## Introduction
This fork is an slightly modification to be able to train the large model in the **Squad 2.0** dataset using a **RTX 2080 (8GB) GPU**.

The modifications are:
- Use FP-16
- Reduce batch_size to 4
- Reduce seq_len to 340.
- Train half of the network, ie, layers 12, 13..., 23. Freeze the others (1, 2, ... 11)
- Replace the FC layers (1024 -> 1) to a deeper FC layer (512 -> 256 -> 1) for start_logits, end_logits and CLS.

The files changed are:
- [scripts/gpu_squad_base_GPU.sh](https://github.com/renatoviolin/xlnet/blob/master/scripts/gpu_squad_base_GPU.sh)
- [run_squad_GPU.py](https://github.com/renatoviolin/xlnet/blob/master/run_squad_GPU.py)
- [model_utils_GPU.py](https://github.com/renatoviolin/xlnet/blob/master/model_utils_GPU.py)
- [function_builder_GPU.py](https://github.com/renatoviolin/xlnet/blob/master/function_builder_GPU.py)

With those modifications I could achieve **86,25 F1-Score** on the **Squad-2.0 dev_set**, training for 85000 steps (~ 3 epochs of the full dataset). This training took about 5-6 hours.

I consider a very good result, since it is trained in a very limited hardware. 

For those who has TPU access, could use the original implementation, traing all the layers, replacing the single FC Layer for a deeper FC layer and see how it improves the network.

