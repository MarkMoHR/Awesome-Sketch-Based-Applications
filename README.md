# Awesome-Sketch-Based-Applications

A collection of sketch based applications.

> Feel free to create a PR or an issue.


**Outline**

- [1. Sketch Based Image Synthesis](#1-sketch-based-image-synthesis)
  - [1.1 Automatic Synthesis](#11-automatic-synthesis)
  - [1.2 Style-based Synthesis](#12-style-based-synthesis)
- [2. Sketch Based Image Editing](#2-sketch-based-image-editing)
- [3. Sketch Based Image Retrieval (SBIR)](#3-sketch-based-image-retrieval-sbir)
- [4. Sketch Based 3D Shape Retrieval](#4-sketch-based-3d-shape-retrieval)
- [5. Sketch Based 3D Shape Modeling](#5-sketch-based-3d-shape-modeling)
- [6. Sketch Completion](#6-sketch-completion)
- [7. Sketch Segmentation and Perceptual Grouping](#7-sketch-segmentation-and-perceptual-grouping)
- [8. Sketch and Shading](#8-sketch-and-shading)
- [9. Sketch-Guided Object Localization](#9-sketch-guided-object-localization)
- [10. Sketch-Guided Video Synthesis](#10-sketch-guided-video-synthesis)
- [11. Sketch Enhancement](#11-sketch-enhancement)
- [12. Sketch Representation Learning](#12-sketch-representation-learning)


---

## 1. Sketch Based Image Synthesis

### 1.1 Automatic Synthesis

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_SketchyGAN_Towards_Diverse_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/wchen342/SketchyGAN) |
| [Image Generation from Sketch Constraint Using Contextual GAN](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yongyi_Lu_Image_Generation_from_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/elliottwu/sText2Image) [[project]](https://elliottwu.com/projects/sketch/) |
| [Multi-Instance Sketch to Image Synthesis With Progressive Generative Adversarial Networks](https://ieeexplore.ieee.org/abstract/document/8698864) | IEEE Access 2019 |  |
| [Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ghosh_Interactive_Sketch__Fill_Multiclass_Sketch-to-Image_Translation_ICCV_2019_paper.pdf) | ICCV 2019 |  [[code]](https://github.com/arnabgho/iSketchNFill) [[project]](https://arnabgho.github.io/iSketchNFill/) |
| [LinesToFacePhoto: Face Photo Generation from Lines with Conditional Self-Attention Generative Adversarial Network](https://arxiv.org/pdf/1910.08914.pdf) | ACM MM 2019 |  |
| [SketchyCOCO: Image Generation from Freehand Scene Sketches](http://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_SketchyCOCO_Image_Generation_From_Freehand_Scene_Sketches_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/sysu-imsl/SketchyCOCO) |
| [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf) | SIGGRAPH 2020 | [[project]](http://geometrylearning.com/DeepFaceDrawing/) [[code]](https://github.com/IGLICT/DeepFaceDrawing-Jittor) |
| [Unsupervised Sketch-to-Photo Synthesis](https://arxiv.org/abs/1909.08313v3) | ECCV 2020 |  [[code]](https://github.com/rt219/Unpaired-Sketch-to-Photo-Translation) [[project]](http://sketch.icsi.berkeley.edu/) |
| [DeepFacePencil: Creating Face Images from Freehand Sketches](https://arxiv.org/abs/2008.13343) | ACM MM 2020 |  |
| [Creative Sketch Generation](https://arxiv.org/abs/2011.10039) | ICLR 2021 | [[code]](https://github.com/facebookresearch/DoodlerGAN) [[project]](http://doodlergan.cloudcv.org/) |

### 1.2 Style-based Synthesis

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [TextureGAN: Controlling Deep Image Synthesis with Texture Patches](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_TextureGAN_Controlling_Deep_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/janesjanes/Pytorch-TextureGAN) |
| [Multimodal Unsupervised Image-to-Image Translation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/nvlabs/MUNIT) |
| [SketchPatch: Sketch Stylization via Seamless Patch-level Synthesis](https://arxiv.org/abs/2009.02216) | SIGGRAPH Asia 2020 |  |


## 2. Sketch Based Image Editing

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [FaceShop: Deep Sketch-based Face Image Editing](https://arxiv.org/pdf/1804.08972.pdf) | SIGGRAPH 2018 | [[project]](https://home.inf.unibe.ch/~porteni/projects/faceshop/) |
| [Sparse, Smart Contours to Represent and Edit Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dekel_Sparse_Smart_Contours_CVPR_2018_paper.pdf) | CVPR 2018 | [[project]](https://contour2im.github.io/) |
| [Example-Guided Style-Consistent Image Synthesis from Semantic Labeling](https://arxiv.org/pdf/1906.01314) | CVPR 2019 | [[code]](https://github.com/cxjyxxme/pix2pixSC) |
| [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589) (DeepFill v2) | ICCV 2019 | [[project]](http://jiahuiyu.com/deepfill) [[code]](https://github.com/JiahuiYu/generative_inpainting) |
| [SC-FEGAN: Face Editing Generative Adversarial Network With User's Sketch and Color](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jo_SC-FEGAN_Face_Editing_Generative_Adversarial_Network_With_Users_Sketch_and_ICCV_2019_paper.pdf) | ICCV 2019 | [[code]](https://github.com/run-youngjoo/SC-FEGAN) |
| [Fashion Editing with Multi-scale Attention Normalization](https://arxiv.org/pdf/1906.00884.pdf) | 1906.00884 |  |
| [Intuitive, Interactive Beard and Hair Synthesis with Generative Models](https://arxiv.org/abs/2004.06848) | CVPR 2020 |  |
| [Deep Plastic Surgery: Robust and Controllable Image Editing with Human-Drawn Sketches](https://arxiv.org/abs/2001.02890) | ECCV 2020 | [[code]](https://github.com/VITA-Group/DeepPS) [[project]](https://williamyang1991.github.io/projects/ECCV2020/) |


## 3. Sketch Based Image Retrieval (SBIR)

- Object-level

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch Me That Shoe](http://openaccess.thecvf.com/content_cvpr_2016/papers/Yu_Sketch_Me_That_CVPR_2016_paper.pdf) | CVPR 2016 | [[code-caffe]](https://github.com/seuliufeng/DeepSBIR) [[code-tf]](https://github.com/yuchuochuo1023/Deep_SBIR_tf) [[project]](http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html) |
| [Deep Multi-task Attribute-driven Ranking for Fine-grained Sketch-based Image Retrieval](http://www.bmva.org/bmvc/2016/papers/paper132/index.html) | BMVC 2016 |  |
| [Deep Sketch Hashing: Fast Free-hand Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Deep_Sketch_Hashing_CVPR_2017_paper.pdf) | CVPR 2017 | [[code]](https://github.com/ymcidence/DeepSketchHashing) |
| [Deep Spatial-Semantic Attention for Fine-Grained Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_ICCV_2017/papers/Song_Deep_Spatial-Semantic_Attention_ICCV_2017_paper.pdf) | ICCV 2017 | [[project]](http://www.eecs.qmul.ac.uk/~js327/Project_pages/Project_iccv2017.html) |
| [Zero-Shot Sketch-Image Hashing](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Zero-Shot_Sketch-Image_Hashing_CVPR_2018_paper.pdf) | CVPR 2018 |  |
| [SketchMate: Deep Hashing for Million-Scale Human Sketch Retrieval](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_SketchMate_Deep_Hashing_CVPR_2018_paper.pdf) | CVPR 2018 |  |
| [Generative Domain-Migration Hashing for Sketch-to-Image Retrieval](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jingyi_Zhang_Generative_Domain-Migration_Hashing_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/YCJGG/GDH) |
| [A Zero-Shot Framework for Sketch Based Image Retrieval](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sasikiran_Yelamarthi_A_Zero-Shot_Framework_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/ShivaKrishnaM/ZS-SBIR) |
| [Generalising Fine-Grained Sketch-Based Image Retrieval](http://www.eecs.qmul.ac.uk/~kp306/Kaiyue%20Material/CVPR_2019/CC_FGSBIR.pdf) | CVPR 2019 |  |
| [Doodle to Search: Practical Zero-Shot Sketch-Based Image Retrieval](https://arxiv.org/pdf/1904.03451.pdf) | CVPR 2019 | [[project]](https://sounakdey.github.io/doodle2search.github.io/) [[code]](https://github.com/sounakdey/doodle2search) |
| [LiveSketch: Query Perturbations for Guided Sketch-Based Visual Search](https://arxiv.org/abs/1904.06611) | CVPR 2019 | |
| [Semantically Tied Paired Cycle Consistency for Zero-Shot Sketch-Based Image Retrieval](https://arxiv.org/abs/1903.03372) | CVPR 2019 | [[code]](https://github.com/AnjanDutta/sem-pcyc) |
| [Generative Model for Zero-Shot Sketch-Based Image Retrieval](https://arxiv.org/abs/1904.08542) | CVPRW 2019 |  |
| [Learning Structural Representations via Dynamic Object Landmarks Discovery for Sketch Recognition and Retrieval](https://ieeexplore.ieee.org/abstract/document/8694004) | TIP 2019 |  |
| [Stacked Semantic-Guided Network for Zero-Shot Sketch-Based Image Retrieval](https://arxiv.org/abs/1904.01971) | 1904.01971 |  |
| [Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Semantic-Aware_Knowledge_Preservation_for_Zero-Shot_Sketch-Based_Image_Retrieval_ICCV_2019_paper.pdf) | ICCV 2019 |  |
| [Semantic Adversarial Network for Zero-Shot Sketch-Based Image Retrieval](https://arxiv.org/abs/1905.02327) | 1905.02327 |  |
| [TC-Net for iSBIR: Triplet Classification Network for Instance-level Sketch Based Image Retrieval](http://www.eecs.qmul.ac.uk/~sgg/papers/LinEtAl_ACM_MM2019.pdf) | ACM MM 2019 |  |
| [Sketch Less for More: On-the-Fly Fine-Grained Sketch Based Image Retrieval](https://arxiv.org/abs/2002.10310) | CVPR 2020 | [[code]](https://github.com/AyanKumarBhunia/on-the-fly-FGSBIR) |
| [Solving Mixed-modal Jigsaw Puzzle for Fine-Grained Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Solving_Mixed-Modal_Jigsaw_Puzzle_for_Fine-Grained_Sketch-Based_Image_Retrieval_CVPR_2020_paper.pdf) | CVPR 2020 |  |
| [Fine-Grained Instance-Level Sketch-Based Image Retrieval](https://link.springer.com/article/10.1007/s11263-020-01382-3) | IJCV 2020 |  |

- Scene-level

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SceneSketcher: Fine-Grained Image Retrieval with Scene Sketches](http://orca.cf.ac.uk/133561/1/SceneSketcherECCV2020.pdf) | ECCV 2020 |  |


## 4. Sketch Based 3D Shape Retrieval

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch-based 3D Shape Retrieval using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Sketch-Based_3D_Shape_2015_CVPR_paper.pdf) | CVPR 2015 |  |
| [Learning Cross-Domain Neural Networks for Sketch-Based 3D Shape Retrieval](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11889) | AAAI 2016 |  |
| [Deep Correlated Metric Learning for Sketch-based 3D Shape Retrieval](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14431) | AAAI 2017 |  |
| [Learning Barycentric Representations of 3D Shapes for Sketch-based 3D Shape Retrieval](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Learning_Barycentric_Representations_CVPR_2017_paper.pdf) | CVPR 2017 |  |
| [Deep Cross-modality Adaptation via Semantics Preserving Adversarial Learning for Sketch-based 3D Shape Retrieval](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jiaxin_Chen_Deep_Cross-modality_Adaptation_ECCV_2018_paper.pdf) | ECCV 2018 |  |
| [Unsupervised Learning of 3D Model Reconstruction from Hand-Drawn Sketches](https://dl.acm.org/citation.cfm?id=3240699) | ACMMM 2018 |  |
| [Towards 3D VR-Sketch to 3D Shape Retrieval](https://rowl1ng.com/assets/pdf/3DV_VRSketch.pdf) | 3DV 2020 | [[code]](https://github.com/ygryadit/Towards3DVRSketch) [[project]](https://rowl1ng.com/projects/3DSketch3DV/) |


## 5. Sketch Based 3D Shape Modeling

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks](https://people.cs.umass.edu/~zlun/papers/SketchModeling/SketchModeling.pdf) | 3DV 2017 | [[code]](https://github.com/happylun/SketchModeling) [[project]](https://people.cs.umass.edu/~zlun/SketchModeling/) |
| [Shape Synthesis from Sketches via Procedural Models and Convolutional Networks](https://people.cs.umass.edu/~kalo/papers/shapepmconvnet/shapepmconvnet.pdf) | TVCG 2017 |  |
| [DeepSketch2Face: A Deep Learning Based Sketching System for 3D Face and Caricature Modeling](https://i.cs.hku.hk/~xghan/papers/deepske2face.pdf) | SIGGRAPH 2017 | [[project]](https://i.cs.hku.hk/~xghan/Projects/ske2face.htm) |
| [BendSketch: Modeling Freeform Surfaces Through 2D Sketching](http://haopan.github.io/papers/bendsketch.pdf) | SIGGRAPH 2017 | [[project]](http://haopan.github.io/bendsketch.html) |
| [Robust Flow-Guided Neural Prediction for Sketch-Based Freeform Surface Modeling](http://haopan.github.io/papers/SketchCNN.pdf) | SIGGRAPH Asia 2018 | [[code]](https://github.com/Enigma-li/SketchCNN) [[project]](http://haopan.github.io/sketchCNN.html) |
| [Learning a Shared Shape Space for Multimodal Garment Design](https://arxiv.org/abs/1806.11335) | SIGGRAPH Asia 2018 |  [[project]](http://geometry.cs.ucl.ac.uk/projects/2018/garment_design/) |
| [DeepSketchHair: Deep Sketch-based 3D Hair Modeling](https://arxiv.org/abs/1908.07198) | TVCG 2019 |  |
| [Lifting Freehand Concept Sketches into 3D](https://repo-sam.inria.fr/d3/Lift3D/Gryaditskaya_SigAsia20_Lifting%20_Freehand_Concept_Sketches_into_3D.pdf) | SIGGRAPH Asia 2020 | [[project]](https://ns.inria.fr/d3/Lift3D/) [[code]](https://github.com/ygryadit/LiftConceptSketches3D) |
| [Sketch2CAD: Sequential CAD Modeling by Sketching in Context](http://geometry.cs.ucl.ac.uk/projects/2020/sketch2cad/paper_docs/Sketch2CAD_SIGA_2020.pdf) | SIGGRAPH Asia 2020 | [[project]](http://geometry.cs.ucl.ac.uk/projects/2020/sketch2cad/) [[code]](https://github.com/Enigma-li/Sketch2CAD) |
| [Interactive Liquid Splash Modeling by User Sketches](https://dl.acm.org/doi/abs/10.1145/3414685.3417832) | SIGGRAPH Asia 2020 |  |
| [Monster Mash: A Single-View Approach to Casual 3D Modeling and Animation](https://dcgi.fel.cvut.cz/home/sykorad/Dvoroznak20-SA.pdf) | SIGGRAPH Asia 2020 | [[project]](https://dcgi.fel.cvut.cz/home/sykorad/monster_mash) [[demo]](http://monstermash.zone/) |


## 6. Sketch Completion

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Joint Gap Detection and Inpainting of Line Drawings](http://iizuka.cs.tsukuba.ac.jp/projects/inpainting/data/inpainting_cvpr2017.pdf) | CVPR 2017 | [[project]](http://iizuka.cs.tsukuba.ac.jp/projects/inpainting/en/) [[code]](https://github.com/kaidlc/CVPR2017_linedrawings) |
| [SketchGAN: Joint Sketch Completion and Recognition with Generative Adversarial Network](https://orca-mwe.cf.ac.uk/121532/1/SketchGAN_CVPR2019.pdf) | CVPR 2019 |  |
| [Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt](https://arxiv.org/abs/2005.09159) | CVPR 2020 |  |


## 7. Sketch Segmentation and Perceptual Grouping

- Sketch Segmentation

<table>
  <tr>
    <td><strong>Type</strong></td>
    <td><strong>Paper</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Code/Project Link</strong></td>
  </tr>
  <tr>
    <td rowspan=3"><strong>Pixelwise</strong></td>
    <td> <a href="https://ieeexplore.ieee.org/abstract/document/8565976">Fast Sketch Segmentation and Labeling With Deep Learning</a> </td> 
    <td> CGA 2019 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://ieeexplore.ieee.org/abstract/document/8784880">SPFusionNet: Sketch Segmentation Using Multi-modal Data Fusion</a> </td> 
    <td> ICME 2019 </td> 
    <td>  </td>
  </tr>
  <tr>
    <td> <a href="http://openaccess.thecvf.com/content_ECCV_2018/papers/Changqing_Zou_SketchyScene_Richly-Annotated_Scene_ECCV_2018_paper.pdf">SketchyScene: Richly-Annotated Scene Sketches</a> (scene-level) </td> 
    <td> ECCV 2018 </td> 
    <td> <a href="https://github.com/SketchyScene/SketchyScene">[code]</a> </td>
  </tr>
  
  <tr>
    <td rowspan="5"><strong>Stroke-level</strong></td>
    <td> <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/16-2012-eccv-sketch-segmenation.pdf">Free Hand-Drawn Sketch Segmentation</a> </td> 
    <td> ECCV 2012 </td> 
    <td>  </td>
  </tr>
  <tr>
    <td> <a href="https://scholars.cityu.edu.hk/files/15839162/9_5915645_TOG_Data_driven_postprint.pdf">Data-driven Segmentation and Labeling of Freehand Sketches</a> </td> 
    <td> SIGGRAPH Asia 2014 </td> 
    <td> <a href="http://sweb.cityu.edu.hk/hongbofu/projects/SketchSegmentationLabeling_SA14/src_global_interpretation.zip">[code]</a> <a href="http://sweb.cityu.edu.hk/hongbofu/projects/SketchSegmentationLabeling_SA14">[project]</a> <a href="http://sweb.cityu.edu.hk/hongbofu/projects/SketchSegmentationLabeling_SA14/Sketch_dataset.zip">[dataset]</a>  </td>
  </tr>
  <tr>
    <td> <a href="http://homes.esat.kuleuven.be/~konijn/publications/2016/a151-schneider.pdf">Example-Based Sketch Segmentation and Labeling Using CRFs</a> </td> 
    <td> TOG 2016 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://ieeexplore.ieee.org/abstract/document/8766108">SketchSegNet+: An End-to-End Learning of RNN for Multi-Class Sketch Semantic Segmentation</a> </td> 
    <td> IEEE Access 2019 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/pdf/1901.03427.pdf">Stroke-based sketched symbol reconstruction and segmentation</a> </td> 
    <td> 1901.03427 </td> 
    <td> </td>
  </tr>
</table>

- Sketch Perceptual Grouping

<table>
  <tr>
    <td><strong>Type</strong></td>
    <td><strong>Paper</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Code/Project Link</strong></td>
  </tr>
  
  <tr>
    <td rowspan="3"><strong>Stroke-level</strong></td>
    <td> <a href="https://ieeexplore.ieee.org/abstract/document/6738056">Sketching by perceptual grouping</a> </td> 
    <td> ICIP 2013 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Qi_Making_Better_Use_2015_CVPR_paper.pdf">Making Better Use of Edges via Perceptual Grouping</a> </td> 
    <td> CVPR 2015 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="http://openaccess.thecvf.com/content_ECCV_2018/papers/Ke_LI_Universal_Sketch_Perceptual_ECCV_2018_paper.pdf">Universal Sketch Perceptual Grouping</a> / <a href="https://ieeexplore.ieee.org/abstract/document/8626530">Toward Deep Universal Sketch Perceptual Grouper</a> </td> 
    <td> ECCV 2018 / TIP 2019 </td> 
    <td> <a href="https://github.com/KeLi-SketchX/Universal-sketch-perceptual-grouping">[code]</a> </td>
  </tr>
  
</table>



## 8. Sketch and Shading

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Learning to Shadow Hand-drawn Sketches](https://arxiv.org/abs/2002.11812) | CVPR 2020 | [[project]](https://cal.cs.umbc.edu/Papers/Zheng-2020-Shade/index.html) [[code]](https://github.com/qyzdao/ShadeSketch) |


## 9. Sketch-Guided Object Localization 

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch-Guided Object Localization in Natural Images](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510528.pdf) | ECCV 2020 | [[project]](http://visual-computing.in/sketch-guided-object-localization/) [[code]](https://github.com/IISCAditayTripathi/SketchGuidedLocalization) |

## 10. Sketch-Guided Video Synthesis

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Deep Sketch-guided Cartoon Video Synthesis](https://arxiv.org/abs/2008.04149) | arxiv |  |

## 11. Sketch Enhancement

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SketchMan: Learning to Create Professional Sketches](https://dl.acm.org/doi/abs/10.1145/3394171.3413720) | ACM MM 2020 | [[code]](https://github.com/LCXCUC/SketchMan2020) |


## 12. Sketch Representation Learning

- Stroke order importance/saliency

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [Pixelor: a competitive sketching AI agent. So you think you can sketch?](https://dl.acm.org/doi/pdf/10.1145/3414685.3417840) | SIGGRAPH Asia 2020 | [[Project]](http://sketchx.ai/pixelor) [[Code]](https://github.com/dasayan05/neuralsort-siggraph) | Vector/stroke-level | Sketch synthesis and recognition |
| [Learning Deep Sketch Abstraction](http://openaccess.thecvf.com/content_cvpr_2018/papers/Muhammad_Learning_Deep_Sketch_CVPR_2018_paper.pdf) | CVPR 2018 |  | Vector/stroke-level | FG-SBIR |

- Representation Learning

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [CoSE: Compositional Stroke Embeddings](https://papers.nips.cc/paper/2020/file/723e8f97fde15f7a8d5ff8d558ea3f16-Paper.pdf) | NeurIPS 2020 | [[Code]](https://github.com/eth-ait/cose) | Vector/stroke-level | Auto-completing diagrams |
| [Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt](https://arxiv.org/abs/2005.09159) | CVPR 2020 |  | Vector/stroke-level | Sketch recognition, retrieval, and gestalt |
| [Sketchformer: Transformer-based Representation for Sketched Structure](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ribeiro_Sketchformer_Transformer-Based_Representation_for_Sketched_Structure_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/leosampaio/sketchformer) | Vector/stroke-level | Sketch classification, SBIR, reconstruction and interpolation |
| [BézierSketch: A generative model for scalable vector sketches](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710630.pdf) | ECCV 2020 | [[project]](https://dasayan05.github.io/pubs/2020/05/22/pub-7.html) [[code]](https://github.com/dasayan05/stroke-ae) | Vector/stroke-level | Scalable sketch generation |

- Pixel-level description and correspondence

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [SketchDesc: Learning Local Sketch Descriptors for Multi-view Correspondence](http://sweb.cityu.edu.hk/hongbofu/doc/SketchDesc_TCSVT2020.pdf) | TCSVT 2020 |  | Pixel-level | Semantic correspondence among multi-view sketches |
| [SketchTransfer: A Challenging New Task for Exploring Detail-Invariance and the Abstractions Learned by Deep Networks](https://arxiv.org/pdf/1912.11570.pdf) | WACV 2020 |  | Pixel-level | Domain transfer learning |
