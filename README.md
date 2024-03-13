# Multi-Scale Implicit Transformer with Re-parameterize  for Arbitrary-Scale Super-Resolution
[Jinchen Zhu](https://github.com/Jinchen2028), Mingjian Zhang, Ling Zheng^, Shizhuang Weng^

Recently, the methods based on implicit neural representations have shown excellent capabilities for arbitrary-scale super-resolution (ASSR). Although these methods represent the features of an image by generating latent codes, these latent codes are difficult to adapt for different magnification factors of super-resolution, which seriously affects their performance. Addressing this, we design Multi-Scale Implicit Transformer (MSIT), consisting of an Multi-scale Neural Operator (MSNO) and Multi-Scale Self-Attention (MSSA). Among them, MSNO obtains multi-scale latent codes through feature enhancement, multi-scale information extraction, and multi-scale information merging. MSSA further enhances the multi-scale characteristics of latent codes, resulting in better performance. Furthermore, to improve the performance of network without increasing the complexity, we propose the Re-Interaction Module (RIM) combined with the cumulative training strategy to enhance the connection of the network parameters. We have systematically introduced multi-scale characteristics for the first time in ASSR task, extensive experiments are performed to validate the effectiveness of MSIT, and our method achieves state-of-the-art performance in arbitrary super-resolution tasks.

> ^: corresponding author(s)
## Dependencies & Installation
```shell
conda create -n msit python
conda activate msit
pip install torch torchvision einops timm matplotlib
```
## Demo
```shell
python demo.py -input ./data/Set5/baby.png -model ./modelzoo/RDN_MSIT.pth -resolution h,w -output test.png
```
## Test
```shell
python test.py -config ./configs/test/test-set5-4.yaml -model ./modelzoo/RDN_MSIT.pth -name RDN_AMSIT
```
## Training
```shell
python train.py -config ./configs/train/train_rdn_baseline_MsIT.yaml -name RDN_MSIT
```

## Citation

If MSIT helps your research or work, please consider citing the following works:

----------
```BibTex
@article{zhu2024mutil,
  title={Multi-Scale Implicit Transformer with Re-parameterize for Arbitrary-Scale Super-Resolution},
  author={Zhu, Jinchen and Zhang, Mingjian and Zheng, Ling and Weng, Shizhuang},
  journal={arXiv preprint arXiv:2403.06536},
  year={2024}
}
```
