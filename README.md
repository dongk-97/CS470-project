# CS470-project
Virtual-try-on with clothes cropping
## Dataset and reference

[[Dataset Partition Label]](https://drive.google.com/open?id=1Jt9DykVUmUo5dzzwyi4C_1wmWgVYsFDl)  [[Sample Try-on Video]](https://www.youtube.com/watch?v=BbKBSfDBcxI) [[Checkpoints]](https://drive.google.com/file/d/1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx/view?usp=sharing) 

[[Dataset_Test]](https://drive.google.com/file/d/1tE7hcVFm8Td8kRh5iYRBSDFdvZIkbUIR/view?usp=sharing) [[Dataset_Train]](https://drive.google.com/file/d/1lHNujZIq6KVeGOOdwnOXVCSR5E7Kv6xv/view?usp=sharing)

[[Paper]](https://arxiv.org/abs/2003.05863)

## Pre-trained model
[[Deep Fahion pre-trained]](https://drive.google.com/file/d/1JN0pAuBdO9ZaYTYhetZkcA-kwspHxBxc/view?usp=sharing)
[[Vgg19 pre-trained]](https://drive.google.com/file/d/1yaylECQa4JyKtmwIZwZGwssmU3lzT8EN/view?usp=sharing)

# Our trained model(clothes masking)
[[Trained model]](https://drive.google.com/file/d/1Xyqi0hTNhq6S07zQuGYmX6aa9cPGVpaV/view?usp=sharing)

## Inference
For Virtual-try-on,
```bash
cd DeepFashion_Try_On
python main.py --phase test
```
For Clothes masking & cropping
```bash
cd Cloth_mask
python test.py
```

## Sample Cloth mask Results

![result1_0](https://user-images.githubusercontent.com/52373758/101453013-fd60cf80-3971-11eb-9e10-20566d374482.jpg)

## Sample Try-on Results
  
![image](https://github.com/dongk-97/CS470-project/Cloth_mask/test_results/011924_0.jpg)

## Training Details
For better inference performance, model G and G2 should be trained with 200 epoches, while model G1 and U net should be trained with 20 epoches.
For Cloth mask, training with 100 epoches works well.
## License
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.


## Dataset
**VITON Dataset** This dataset is presented in [VITON](https://github.com/xthan/VITON), containing 19,000 image pairs, each of which includes a front-view woman image and a top clothing image. After removing the invalid image pairs, it yields 16,253 pairs, further splitting into a training set of 14,221 paris and a testing set of 2,032 pairs.
