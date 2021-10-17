# [Learning Synergistic Attention for Light Field Salient Object Detection (BMVC 2021)](https://arxiv.org/abs/2104.13916)

Authors: [*Yi Zhang*](https://scholar.google.com/citations?user=NeHBHVUAAAAJ&hl=en), [*Geng Chen*](https://scholar.google.com/citations?user=sJGCnjsAAAAJ&hl=en), [*Qian Chen*](https://scholar.google.com/citations?user=Wz0lfcwAAAAJ&hl=en), [*YuJia Sun*](https://github.com/thograce), [*Yong Xia*](https://scholar.google.com/citations?user=Usw1jeMAAAAJ&hl=en), [*Olivier Deforges*](https://scholar.google.com/citations?user=c5DiiBUAAAAJ&hl=en), [*Wassim Hamidouche*](https://scholar.google.com/citations?user=ywBnUIAAAAAJ&hl=en), [*Lu Zhang*](https://luzhang.perso.insa-rennes.fr/)

# Introduction

<p align="center">
    <img src="./Figures/fig_architecture.jpg" width="90%"/> <br />
    <em> 
    Figure 1: An overview of our SA-Net. Multi-modal multi-level features extracted from our multi-modal encoder are fed to two cascaded synergistic attention (SA) modules followed by a progressive fusion (PF) module. The short names in the figure are detailed as follows: CoA = co-attention component. CA = channel attention component. AA = AiF-induced attention component. RB = residual block. Pn = the nth saliency prediction. (De)Conv = (de-)convolutional layer. BN = batch normalization layer. FC = fully connected layer.
    </em>
</p>

In this work, we propose Synergistic Attention Network (SA-Net) to address the light field salient object detection by establishing a synergistic effect between multimodal features with advanced attention mechanisms. Our SA-Net exploits the rich information of focal stacks via 3D convolutional neural networks, decodes the high-level features of multi-modal light field data with two cascaded synergistic attention modules, and predicts the saliency map using an effective feature fusion module in a progressive manner. Extensive experiments on three widely-used benchmark datasets show that our SA-Net outperforms 28 state-of-the-art models, sufficiently demonstrating its effectiveness and superiority.
