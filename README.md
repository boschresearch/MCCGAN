# Multi-Class Multi-Instance Count Conditioned Adversarial Image Generation MCC-StyleGAN2 

Implementation of the ICCV 2021 paper "Multi-Class Multi-Instance Count Conditioned Adversarial Image Generation MCC-StyleGAN2". The paper can be found 
[here](https://openaccess.thecvf.com/content/ICCV2021/papers/Saseendran_Multi-Class_Multi-Instance_Count_Conditioned_Adversarial_Image_Generation_ICCV_2021_paper.pdf). The code allows the users to
reproduce and extend the results reported in the paper. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication. It will neither be
maintained nor monitored in any way.

## Setup.

### MCC-SimpleGAN
This repository is a PyTorch implementation of a simple convolution based architecture with the same concept for toy experiments.

### Requirements
Please add the project folder to PYTHONPATH and install the required dependencies:
### Dependencies
- python 3.6.10
- pytorch 1.4.0

```
pip install -r requirements.txt
```

### Training

To train the model, run this command:

```train
python main.py --input_images <path_to_image_array> --countvec_path <path_to_count_csvfile> 
```

### MCC-StyleGAN2 
For MCC-StyleGAN2 repository we adapted the official tensorflow implementation of StyleGAN2 "https://github.com/NVlabs/stylegan2". 
For CityCount images, adaptive discriminator augmentation technique based on the implementation in "https://github.com/NVlabs/stylegan2-ada" is utilized while training. Please note that the network configuration and the loss functions however, remains the same for all datasets.
The main differences are in the network architecture and the loss functions used in the training.

training/networks_stylegan2.py - The modified generator and discriminator for count conditioned image generation. To be precise we introduced count vector mapping to each layer in the mapping network. We also introduced dense connectivity where output from a layer is connected to all its following layers.
```
   def denseblock(x, resolution):
        dense = [None] * 5
        if (resolution > 2 and resolution < 7):
            dense[resolution - 3] = x
            
        if resolution - 3 > 0:
           #Denseskip0
                dense[0] = conv2dlayer(dense[0])
        if resolution - 4 > 0:
            #Denseskip1
                dense[1] = conv2dlayer(dense[1])
        if resolution - 5 > 0:
            #Denseskip2
                dense[2] = conv2dlayer(dense[2])
        if resolution - 6 > 0:
           #Denseskip3
                dense[3] = conv2dlayer(dense[3])
        if resolution - 7 > 0:
            #Denseskip4
                dense[4] = conv2dlayer(dense[4])

        if resolution > 3:
            #Denseskipx
                dense[resolution - 4] = conv2dlayer(dense[resolution - 4])
                for iter in range(0, resolution - 3):
                    x = x + dense[iter]
                x = x * (1 / np.sqrt(max(2, resolution - 3)))
        return x
```
training/loss.py - An additional count loss functions used for training the network 
```
countlossfake = tf.reduce_mean(tf.squared_difference(fake_count_out, count_labels))
countlossreal = tf.reduce_mean(tf.squared_difference(real_count_out, count_labels))
countloss = (countlossfake + countlossreal) / 2
```

Incorporate these additional files provided in the MCC-StyeGAN2 directory to the corresponding folders in the original repo.


## Dataset
- To create MultiMNIST dataset, please refer to the repo [here.](https://github.com/shaohua0116/MultiDigitMNIST)

- To create CLEVR dataset, please refer to the CLEVR directory adapted from the [repo](https://github.com/facebookresearch/clevr-dataset-gen).
Incorporate the additional files provided in the CLEVR directory to the corresponding folders in the original repo.
The CLEVR directory includes image generation scripts specifically modified for CLEVR2 and CLEVR3 with multiprocessing enabled.
Refer to the original repo README for instructions on running the image generation.

- To create CityCount dataset, please refer to the Citycount directory. Please download the required dataset(leftImg8bit_trainvaltest.zip, gtBbox3d_trainvaltest.zip and gtBbox_cityPersons_trainval.zip) from [here.](https://www.cityscapes-dataset.com/downloads/)


## License

MCC-StyleGAN2 is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.
For a list of other open source components included in the MCC-StyleGAN2, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).
For further queries, please contact amrutha.saseendran21@gmail.com
