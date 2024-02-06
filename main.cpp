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

int main(int argc,char * argv[])
{
  //- get controls
  Dictionary controlDict = Dictionary("../debug.txt");
  bool debug = controlDict.get<bool>("DEBUG");
  bool infer = controlDict.get<bool>("INFER");
  bool plotInitial_condition = controlDict.get<bool>("PLOT_INITIAL_CONDITION");
  
  std::cout<<"debug mode is set to: "<<debug<<"\n";
  
  // check if CUDA is avalilable and train on GPU if yes
  auto cuda_available = torch::cuda::is_available();
  auto device_str = cuda_available ? torch::kCUDA : torch::kCPU;
  
  //- set device 
  torch::Device device(device_str);
  std::cout << (cuda_available ? "CUDA available. Training on GPU.\n" : "Training on CPU.\n") << '\n';
  
  Dictionary netDict = Dictionary("../params.txt");
  Dictionary meshDict = Dictionary("../mesh.txt");
  float t_min = meshDict.get<float>("lbT");
  std::cout<<"lower time bound is: "<<t_min<<std::endl;
  Dictionary thermoDict = Dictionary("../thermo.txt");
  
  //- create nets
  auto net1 = PinNet(netDict);
  auto net2 = PinNet(netDict);
  
  //- if training from a checkPoint and previously saved neural network is available,
  //- load that neural network 
  if(t_min != 0)
  {
    std::cout<<"training from a checkPoint, loading previously converged neural net:\n"<<std::endl;
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
      std::cerr<< "Error loading model!\n "<<e.what()<<std::endl;
      throw::std::runtime_error("failed to load previously converged neural net\n");
    }
  }
  
  net1->to(device);
  net2->to(device);
  
  thermoPhysical thermo(thermoDict);
  mesh2D mesh(meshDict,net1,net2,device,thermo);
  
  //- plot time t=0 phasefield contour if needed.
  if(plotInitial_condition)
  {
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
  } 

  //- infer previously saved neural net to get fields at initial condition if needed
  if(infer)
  {
    inference("pNet0.500000.pt", netDict, mesh);
  }
  
  torch::optim::Adam adam_optim1(mesh.net_->parameters(), torch::optim::AdamOptions(1e-3));  
  torch::optim::Adam adam_optim2(mesh.net_->parameters(), torch::optim::AdamOptions(1e-4)); 
  torch::optim::Adam adam_optim3(mesh.net_->parameters(), torch::optim::AdamOptions(1e-5));

  //- Time marching loop
  for(int N=0;N<2;N++)
  {
    int epoch=1;
    float loss;
    
    std::cout<<"Traning...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    //- epoch loop
    while(epoch<=mesh.net_->K_EPOCH)
    {
      auto closure = [&](torch::optim::Optimizer &optim)
      { 
        float totalLoss=0.0;
        for(int i=0;i<mesh.net_->NITER_;i++)
        {
          mesh.update(i);
          auto loss = CahnHillard::loss(mesh);
          loss.backward();
          totalLoss +=loss.item<float>();
        }
        optim.step();
        optim.zero_grad();
        return totalLoss/mesh.net_->NITER_;
      };

      torch::optim::Optimizer * current_optim;
      //- learning rate schedule
      switch (epoch)
      {
        case 1 ... 2:
          loss = closure(adam_optim1);
          current_optim = &adam_optim1;
          break;
        case 3 ... 5:
          loss = closure(adam_optim2);
          current_optim = &adam_optim2;
          break;
        default:
          loss = closure(adam_optim3);
          current_optim = &adam_optim3;
          break;
      }

      if (epoch % 1 == 0) 
      {
        std::cout << "  epoch=" << epoch << ", loss=" << std::setprecision(7) << loss<<" lr: "<<current_optim->defaults().get_lr()<<"\n";
      }

      if(epoch % 5000 == 0)
      {
        std::cout<<"saving output..."<<"\n";
        std::string modelName = "pNetSave" + std::to_string(mesh.ubT_)+".pt";
        torch::Tensor grid = torch::stack
        (
          {
            torch::flatten(mesh.xyGrid[0]),
            torch::flatten(mesh.xyGrid[1]),
            torch::full_like(torch::flatten(mesh.xyGrid[1]),mesh.ubT_) //time values 
          },1
        );
        grid.to(mesh.device_);  
        torch::Tensor output = mesh.net_->forward(grid);
        std::string gridName = "gridSave" + std::to_string(mesh.ubT_);
        std::string fieldsName = "fieldsSave" + std::to_string(mesh.ubT_);
        writeTensorToFile(grid,gridName);
        writeTensorToFile(output,fieldsName);
        torch::save(mesh.net_,modelName);
      }

      if (loss < mesh.net_->ABS_TOL) 
      {
        std::string modelName = "pNet" + std::to_string(mesh.ubT_) + ".pt"; 
        torch::save(mesh.net_,modelName);
        epoch += 1;
        break;
      }
      epoch += 1;
    }
   
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Epoch execution time: " << duration.count() << " microseconds" << std::endl;
    
    torch::Tensor grid = torch::stack
    (
      {
        torch::flatten(mesh.xyGrid[0]),
        torch::flatten(mesh.xyGrid[1]),
        torch::full_like(torch::flatten(mesh.xyGrid[1]),mesh.ubT_) //time values 
      },1
    );
    
    mesh.updateMesh();
    mesh.net_->reset_layers();

    torch::Tensor output = mesh.netPrev_->forward(grid);
    std::string gridName = "grid" + std::to_string(mesh.ubT_);
    std::string fieldsName = "fields" + std::to_string(mesh.ubT_);
    writeTensorToFile(grid,gridName);
    writeTensorToFile(output,fieldsName); 
  }

  return 0;
} 
