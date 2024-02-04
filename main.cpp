// main file, run code here
#include <ATen/core/TensorBody.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/mode.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/cuda.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>
#include "nn_main.h"
#include "utils.h"

//- loads in python like indexing of tensors
using namespace torch::indexing;

//- main 
int main(int argc,char * argv[])
{
  //- get debug mode
  Dictionary debugDict = Dictionary("../debug.txt");
  bool debug = debugDict.get<bool>("DEBUG");
  if(debug)
    std::cout<<"debug is: "<<debug<<"\n";
  
  // check if CUDA is avalilable and train on GPU if yes
  auto cuda_available = torch::cuda::is_available();
  auto device_str = cuda_available ? torch::kCUDA : torch::kCPU;
  
  //- create device reference  
  torch::Device device(device_str);
  //- Info out
  std::cout << (cuda_available ? "CUDA available. Training on GPU.\n" : "Training on CPU.\n") << '\n';
  
  
  //- both nets share the same architecture, only network params update
  Dictionary netDict = Dictionary("../params.txt");
  
  //- create dict for mesh
  Dictionary meshDict = Dictionary("../mesh.txt");
  
  //- get t_min to see if training from 0 or from checkPoint
  float t_min = meshDict.get<float>("lbT");
  //- convert to string to match name of saved pytorch model
  std::cout<<"lower time bound is: "<<t_min<<std::endl;

  //- create dict for thermoPhysical class
  Dictionary thermoDict = Dictionary("../thermo.txt");

  //- create first net primary net, is the one being trained
  auto net1 = PinNet(netDict);
  //- create second neural network identical to net1
  auto net2 = PinNet(netDict);
  
  //- if training from a checkPoint and previously saved neural network is available,
  //- load that neural network 
  if(t_min != 0)
  {
    std::cout<<"training from a checkPoint, loading previous neural net\n"<<std::endl;
    std::string model_name = "pNet" + std::to_string(t_min) + ".pt";
    torch::NoGradGuard no_grad;
    torch::serialize::InputArchive in;
    try 
    {
      in.load_from(model_name);
      net2->load(in);
    }
    catch (const std::exception &e)
    {
      std::cerr<< "Error loading model!\n"<<e.what()<<std::endl;
      throw::std::runtime_error("failed to load previously converged neural net\n");
    }
  }
  
  //- transfer nets to GPU if available
  net1->to(device);
  net2->to(device);
  
  //- create thermoPhysical object
  thermoPhysical thermo(thermoDict);
  
  //- create Mesh
  mesh2D mesh(meshDict,net1,net2,device,thermo);
  
  //- code to save initial condition for the phase field variable 
  mesh.iIC_ = torch::stack
  (
    {
      torch::flatten(mesh.initialGrid_[0]),
      torch::flatten(mesh.initialGrid_[1]),
      torch::flatten(mesh.initialGrid_[2])
    },1
  );
 

  writeTensorToFile(mesh.iIC_,"total.txt");
  torch::Tensor C = CahnHillard::C_at_InitialTime(mesh);
  writeTensorToFile(C,"intial.txt");
    
  if(debug)
  { 
    mesh.update(0);
    mesh.update(1);
    std::cout<<mesh.fieldsPDE_<<std::endl;
  }

  inference("pNet0.500000.pt", netDict, mesh);
  //- TODO intialize optim paras from dictionary to avoid recompilation 
  //- declare optimizer instance to be used in training
  //- learning rate is decreased over the traning process and the optimizer class instance is changed
    torch::optim::Adam adam_optim1(mesh.net_->parameters(), torch::optim::AdamOptions(1e-3));  
    torch::optim::Adam adam_optim2(mesh.net_->parameters(), torch::optim::AdamOptions(1e-4)); 
    torch::optim::Adam adam_optim3(mesh.net_->parameters(), torch::optim::AdamOptions(1e-5));

  //- Time marching loop
  for(int N=0;N<2;N++)
  {
    // torch::optim::Adam adam_optim1(mesh.net_->parameters(), torch::optim::AdamOptions(1e-3));  
    // torch::optim::Adam adam_optim2(mesh.net_->parameters(), torch::optim::AdamOptions(1e-4)); 
    // torch::optim::Adam adam_optim3(mesh.net_->parameters(), torch::optim::AdamOptions(1e-5));


    //- set up epoch loop
    int iter=1;
    //- init place holder for loss
    float loss;

    //- file to print out loss history
    std::cout<<"Traning...\n";
    
    //- start profile clock
    auto start_time = std::chrono::high_resolution_clock::now();
    //- epoch loop
    while(iter<=mesh.net_->K_EPOCH)
    {
      //- define closure for optimizer class to work with
      auto closure = [&](torch::optim::Optimizer &optim)
      { 
        float totalLoss=0.0;
        for(int i=0;i<mesh.net_->NITER_;i++)
        {
          //- generate solution fields from forward pass, accumulate gradients
          mesh.update(i);
          //- get total loss for the optimizer (PDE,IC,BC)
          auto loss = CahnHillard::loss(mesh);
          //- back propogate and accumulate gradiets of loss wrt to parameters
          loss.backward();
          //- get the total loss across all iterations within the epoch
          totalLoss +=loss.item<float>();
        }
        //- update network parameters
        optim.step();
        //- clear gradients for next epoch
        optim.zero_grad();
        //- return the average loss
        return totalLoss/mesh.net_->NITER_;
      };

      //- print out iteration numbers
      if(debug)
      {
        std::cout<<iter<<"\n";
      }
      
      //- learning rate schedule
      if (iter <= 4000)
      {
        loss = closure(adam_optim1);
      } 
      else if(iter <= 8000) 
      { 
        loss = closure(adam_optim2);
      }
      else
      {
        loss = closure(adam_optim3);
      }
      
      //- Print out loss info every 10 epochs
      if (iter % 5 == 0) 
      {
        std::cout << "  iter=" << iter << ", loss=" << std::setprecision(7) << loss<<" lr: "<<adam_optim1.defaults().get_lr()<<"\n";
      }
      if(iter % 5000 == 0)
      {
        std::cout<<"saving output..."<<"\n";
        //- save unconverged model
        std::string modelName = "pNetSave" + std::to_string(mesh.ubT_);
        //- input grid
        torch::Tensor grid = torch::stack
        (
          {
            torch::flatten(mesh.xyGrid[0]),
            torch::flatten(mesh.xyGrid[1]),
            torch::full_like(torch::flatten(mesh.xyGrid[1]),mesh.ubT_) //time values 
          },1
        );
    
        //- transfer grid to device
        grid.to(mesh.device_);  

        //- get predicted output, and from that get phaseField 
        torch::Tensor C1 = mesh.net_->forward(grid);
        std::string gridName = "gridSave" + std::to_string(mesh.ubT_);
        std::string fieldsName = "fieldsSave" + std::to_string(mesh.ubT_);
        //- write out input data for python to plot
        writeTensorToFile(grid,gridName);
        writeTensorToFile(C1,fieldsName); 
   
      }
      //- stop training if target loss achieved
      if (loss < mesh.net_->ABS_TOL) 
      {
        std::string modelName = "pNet" + std::to_string(mesh.ubT_) + ".pt"; // ".pt" is the extension of for pyTorch module
        //- save model to file for post processing, indicate saved model is due to convergence
        torch::save(mesh.net_,modelName);
        //- update iter to get correct iter count
        iter += 1;
        //- update mesh for nex Time step in the time marching loop
        break;
      }
      iter += 1;
    }
    
    //- end profile clock
    auto end_time = std::chrono::high_resolution_clock::now();
    
    //- get runTime for traning
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
   
    //- info out runTime
    std::cout << "Epoch execution time: " << duration.count() << " microseconds" << std::endl;
    
    //- Grid  for plotting final timeStep
    torch::Tensor grid = torch::stack
    (
      {
        torch::flatten(mesh.xyGrid[0]),
        torch::flatten(mesh.xyGrid[1]),
        torch::full_like(torch::flatten(mesh.xyGrid[1]),0.5) //time values 
        //torch::flatten(mesh.mesh_[2])
      },1
    );
    
    // transfer learned paras from net1 to net2
    mesh.updateMesh();
    
    //- reset net1 to create a fresh neural network
    mesh.net_->reset_layers();
    
    //- get predicted output, and from that get phaseField 
    torch::Tensor C1 = mesh.netPrev_->forward(grid);
    
    //- file Names for the
    std::string gridName = "grid" + std::to_string(mesh.ubT_);
    std::string fieldsName = "fields" + std::to_string(mesh.ubT_);

    //- write out input data for python to plot
    writeTensorToFile(grid,gridName);
    writeTensorToFile(C1,fieldsName); 
  }

  return 0;
} 
