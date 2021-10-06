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
- [6. Sketch Based Garment Design](#6-sketch-based-garment-design)
- [7. Sketch Completion](#7-sketch-completion)
- [8. Manga Restoration and Inpainting](#8-manga-restoration-and-inpainting)
- [9. Sketch and Shading](#9-sketch-and-shading)
- [10. Sketch Enhancement](#10-sketch-enhancement)
- [11. Sketch-Guided Object Localization](#11-sketch-guided-object-localization)
- [12. Sketch-Guided Video Synthesis](#12-sketch-guided-video-synthesis)
- [13. Sketch Segmentation and Perceptual Grouping](#13-sketch-segmentation-and-perceptual-grouping)
- [14. Sketch Representation Learning](#14-sketch-representation-learning)
- [15. Sketch Animation/Inbetweening](#15-sketch-animationinbetweening)
- [16. Sketch and AR/VR](#16-sketch-and-arvr)


---

## 1. Sketch Based Image Synthesis

### 1.1 Automatic Synthesis

- Natural Image or Object

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_SketchyGAN_Towards_Diverse_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/wchen342/SketchyGAN) |
| [Image Generation from Sketch Constraint Using Contextual GAN](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yongyi_Lu_Image_Generation_from_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/elliottwu/sText2Image) [[project]](https://elliottwu.com/projects/sketch/) |
| [Multi-Instance Sketch to Image Synthesis With Progressive Generative Adversarial Networks](https://ieeexplore.ieee.org/abstract/document/8698864) | IEEE Access 2019 |  |
| [Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ghosh_Interactive_Sketch__Fill_Multiclass_Sketch-to-Image_Translation_ICCV_2019_paper.pdf) | ICCV 2019 |  [[code]](https://github.com/arnabgho/iSketchNFill) [[project]](https://arnabgho.github.io/iSketchNFill/) |
| [SketchyCOCO: Image Generation from Freehand Scene Sketches](http://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_SketchyCOCO_Image_Generation_From_Freehand_Scene_Sketches_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/sysu-imsl/SketchyCOCO) |
| [Unsupervised Sketch-to-Photo Synthesis](https://arxiv.org/abs/1909.08313v3) | ECCV 2020 |  [[code]](https://github.com/rt219/Unpaired-Sketch-to-Photo-Translation) [[project]](http://sketch.icsi.berkeley.edu/) |
| [Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis](https://arxiv.org/abs/2104.05703) | arxiv 2104 |  |
| [Sketch Your Own GAN](https://arxiv.org/abs/2108.02774) | ICCV 2021 | [[code]](https://github.com/peterwang512/GANSketching) [[webpage]](https://peterwang512.github.io/GANSketching/) |


- Human Face

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [LinesToFacePhoto: Face Photo Generation from Lines with Conditional Self-Attention Generative Adversarial Network](https://arxiv.org/pdf/1910.08914.pdf) | ACM MM 2019 |  |
| [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf) | SIGGRAPH 2020 | [[project]](http://geometrylearning.com/DeepFaceDrawing/) [[code]](https://github.com/IGLICT/DeepFaceDrawing-Jittor) |
| [DeepFacePencil: Creating Face Images from Freehand Sketches](https://arxiv.org/abs/2008.13343) | ACM MM 2020 | [[project]](https://liyuhangustc.github.io/Sketch2Face/) [[code]](https://github.com/LiYuhangUSTC/Sketch2Face) |

### 1.2 Style-based Synthesis

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [TextureGAN: Controlling Deep Image Synthesis with Texture Patches](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_TextureGAN_Controlling_Deep_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/janesjanes/Pytorch-TextureGAN) |
| [Multimodal Unsupervised Image-to-Image Translation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/nvlabs/MUNIT) |
| [SketchPatch: Sketch Stylization via Seamless Patch-level Synthesis](https://arxiv.org/abs/2009.02216) | SIGGRAPH Asia 2020 |  |
| [Self-Supervised Sketch-to-Image Synthesis](https://arxiv.org/abs/2012.09290) | AAAI 2021 | [[code]](https://github.com/odegeasslbc/Self-Supervised-Sketch-to-Image-Synthesis-PyTorch) |


## 2. Sketch Based Image Editing

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [FaceShop: Deep Sketch-based Face Image Editing](https://arxiv.org/pdf/1804.08972.pdf) | SIGGRAPH 2018 | [[project]](https://home.inf.unibe.ch/~porteni/projects/faceshop/) |
| [CaricatureShop: Personalized and Photorealistic Caricature Sketching](https://ieeexplore.ieee.org/document/8580421) | TVCG 2018 |  |
| [Sparse, Smart Contours to Represent and Edit Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dekel_Sparse_Smart_Contours_CVPR_2018_paper.pdf) | CVPR 2018 | [[project]](https://contour2im.github.io/) |
| [Example-Guided Style-Consistent Image Synthesis from Semantic Labeling](https://arxiv.org/pdf/1906.01314) | CVPR 2019 | [[code]](https://github.com/cxjyxxme/pix2pixSC) |
| [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589) (DeepFill v2) | ICCV 2019 | [[project]](http://jiahuiyu.com/deepfill) [[code]](https://github.com/JiahuiYu/generative_inpainting) |
| [SC-FEGAN: Face Editing Generative Adversarial Network With User's Sketch and Color](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jo_SC-FEGAN_Face_Editing_Generative_Adversarial_Network_With_Users_Sketch_and_ICCV_2019_paper.pdf) | ICCV 2019 | [[code]](https://github.com/run-youngjoo/SC-FEGAN) |
| [Fashion Editing with Multi-scale Attention Normalization](https://arxiv.org/pdf/1906.00884.pdf) | 1906.00884 |  |
| [Intuitive, Interactive Beard and Hair Synthesis with Generative Models](https://arxiv.org/abs/2004.06848) | CVPR 2020 |  |
| [Deep Plastic Surgery: Robust and Controllable Image Editing with Human-Drawn Sketches](https://arxiv.org/abs/2001.02890) | ECCV 2020 | [[code]](https://github.com/VITA-Group/DeepPS) [[project]](https://williamyang1991.github.io/projects/ECCV2020/) |
| [DeepFaceEditing: Deep Face Generation and Editing with Disentangled Geometry and Appearance Control](http://www.geometrylearning.com/DeepFaceEditing/) | SIGGRAPH 2021 | [[code]](https://github.com/IGLICT/DeepFaceEditing-Jittor) [[project]](http://www.geometrylearning.com/DeepFaceEditing/) |
| [SketchHairSalon: Deep Sketch-based Hair Image Synthesis](https://arxiv.org/abs/2109.07874) | SIGGRAPH Asia 2021 | [[project]](https://chufengxiao.github.io/SketchHairSalon/) |
| [DeepSIM: Image Shape Manipulation from a Single Augmented Training Sample](https://arxiv.org/abs/2109.06151) | ICCV 2021 | [[code]](https://github.com/eliahuhorwitz/DeepSIM) [[project]](http://www.vision.huji.ac.il/deepsim/) |


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
| [Learning Structural Representations via Dynamic Object Landmarks Discovery for Sketch Recognition and Retrieval](https://ieeexplore.ieee.org/abstract/document/8694004) | TIP 2019 |  |
| [Stacked Semantic-Guided Network for Zero-Shot Sketch-Based Image Retrieval](https://arxiv.org/abs/1904.01971) | 1904.01971 |  |
| [Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Semantic-Aware_Knowledge_Preservation_for_Zero-Shot_Sketch-Based_Image_Retrieval_ICCV_2019_paper.pdf) | ICCV 2019 |  |
| [Semantic Adversarial Network for Zero-Shot Sketch-Based Image Retrieval](https://arxiv.org/abs/1905.02327) | 1905.02327 |  |
| [TC-Net for iSBIR: Triplet Classification Network for Instance-level Sketch Based Image Retrieval](http://www.eecs.qmul.ac.uk/~sgg/papers/LinEtAl_ACM_MM2019.pdf) | ACM MM 2019 |  |
| [Sketch Less for More: On-the-Fly Fine-Grained Sketch Based Image Retrieval](https://arxiv.org/abs/2002.10310) | CVPR 2020 | [[code]](https://github.com/AyanKumarBhunia/on-the-fly-FGSBIR) |
| [Solving Mixed-modal Jigsaw Puzzle for Fine-Grained Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Solving_Mixed-Modal_Jigsaw_Puzzle_for_Fine-Grained_Sketch-Based_Image_Retrieval_CVPR_2020_paper.pdf) | CVPR 2020 |  |
| [Fine-Grained Instance-Level Sketch-Based Image Retrieval](https://link.springer.com/article/10.1007/s11263-020-01382-3) | IJCV 2020 |  |
| [StyleMeUp: Towards Style-Agnostic Sketch-Based Image Retrieval](https://arxiv.org/abs/2103.15706) | CVPR 2021 |  |
| [More Photos are All You Need: Semi-Supervised Learning for Fine-Grained Sketch-Based Image Retrieval](https://arxiv.org/abs/2103.13990) | CVPR 2021 | [[code]](https://github.com/AyanKumarBhunia/semisupervised-FGSBIR) |

- Scene-level

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SceneSketcher: Fine-Grained Image Retrieval with Scene Sketches](http://orca.cf.ac.uk/133561/1/SceneSketcherECCV2020.pdf) | ECCV 2020 |  |

- Video Retrieval

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Fine-Grained Instance-Level Sketch-Based Video Retrieval](https://ieeexplore.ieee.org/abstract/document/9161000) | TCSVT 2020 |  |


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
| [DeepSketch2Face: A Deep Learning Based Sketching System for 3D Face and Caricature Modeling](https://i.cs.hku.hk/~xghan/papers/deepske2face.pdf) | SIGGRAPH 2017 | [[project]](https://i.cs.hku.hk/~xghan/Projects/ske2face.htm) [[code]](https://github.com/changgyhub/deepsketch2face) |
| [BendSketch: Modeling Freeform Surfaces Through 2D Sketching](http://haopan.github.io/papers/bendsketch.pdf) | SIGGRAPH 2017 | [[project]](http://haopan.github.io/bendsketch.html) |
| [Robust Flow-Guided Neural Prediction for Sketch-Based Freeform Surface Modeling](http://haopan.github.io/papers/SketchCNN.pdf) | SIGGRAPH Asia 2018 | [[code]](https://github.com/Enigma-li/SketchCNN) [[project]](http://haopan.github.io/sketchCNN.html) |
| [DeepSketchHair: Deep Sketch-based 3D Hair Modeling](https://arxiv.org/abs/1908.07198) | TVCG 2019 |  |
| [Lifting Freehand Concept Sketches into 3D](https://repo-sam.inria.fr/d3/Lift3D/Gryaditskaya_SigAsia20_Lifting%20_Freehand_Concept_Sketches_into_3D.pdf) | SIGGRAPH Asia 2020 | [[project]](https://ns.inria.fr/d3/Lift3D/) [[code]](https://github.com/ygryadit/LiftConceptSketches3D) |
| [Sketch2CAD: Sequential CAD Modeling by Sketching in Context](http://geometry.cs.ucl.ac.uk/projects/2020/sketch2cad/paper_docs/Sketch2CAD_SIGA_2020.pdf) | SIGGRAPH Asia 2020 | [[project]](http://geometry.cs.ucl.ac.uk/projects/2020/sketch2cad/) [[code]](https://github.com/Enigma-li/Sketch2CAD) |
| [Interactive Liquid Splash Modeling by User Sketches](https://dl.acm.org/doi/abs/10.1145/3414685.3417832) | SIGGRAPH Asia 2020 |  |
| [Monster Mash: A Single-View Approach to Casual 3D Modeling and Animation](https://dcgi.fel.cvut.cz/home/sykorad/Dvoroznak20-SA.pdf) | SIGGRAPH Asia 2020 | [[project]](https://dcgi.fel.cvut.cz/home/sykorad/monster_mash) [[code]](https://github.com/google/monster-mash) [[demo]](http://monstermash.zone/) |
| [SAniHead: Sketching Animal-like 3D Character Heads Using a View-surface Collaborative Mesh Generative Network](http://sweb.cityu.edu.hk/hongbofu/doc/2020TVCG_SAniHead.pdf) | TVCG 2020 |  |
| [Sketch2Model: View-Aware 3D Modeling from Single Free-Hand Sketches](https://arxiv.org/abs/2105.06663) | CVPR 2021 |  |
| [Sketch2Mesh: Reconstructing and Editing 3D Shapes from Sketches](https://arxiv.org/abs/2104.00482v1) | ICCV 2021 |  |


## 6. Sketch Based Garment Design

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketching Folds: Developable Surfaces from Non-Planar Silhouettes](http://imagecomputing.net/damien.rohmer/publications/2015_tog_sketching_folds/article/JungHRBBC_TOG_2015_sketching_folds.pdf) | TOG 2015 |  [[project]](http://imagecomputing.net/damien.rohmer/publications/2015_tog_sketching_folds/index.html) |
| [FoldSketch: Enriching Garments with Physically Reproducible Folds](http://www.cs.ubc.ca/labs/imager/tr/2018/FoldSketch/doc/FoldSketch.pdf) | SIGGRAPH 2018 |  [[project]](http://www.cs.ubc.ca/labs/imager/tr/2018/FoldSketch/) |
| [Learning a Shared Shape Space for Multimodal Garment Design](https://arxiv.org/abs/1806.11335) | SIGGRAPH Asia 2018 |  [[project]](http://geometry.cs.ucl.ac.uk/projects/2018/garment_design/) |


## 7. Sketch Completion

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt](https://arxiv.org/abs/2005.09159) | CVPR 2020 |  |
| [SketchHealer: A Graph-to-Sequence Network for Recreating Partial Human Sketches](https://core.ac.uk/download/pdf/334949144.pdf) | BMVC 2020 | [[code]](https://github.com/sgybupt/SketchHealer) | 
| [SketchGAN: Joint Sketch Completion and Recognition with Generative Adversarial Network](https://orca-mwe.cf.ac.uk/121532/1/SketchGAN_CVPR2019.pdf) | CVPR 2019 |  |
| [Joint Gap Detection and Inpainting of Line Drawings](http://iizuka.cs.tsukuba.ac.jp/projects/inpainting/data/inpainting_cvpr2017.pdf) | CVPR 2017 | [[project]](http://iizuka.cs.tsukuba.ac.jp/projects/inpainting/en/) [[code]](https://github.com/kaidlc/CVPR2017_linedrawings) |


## 8. Manga Restoration and Inpainting

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Exploiting Aliasing for Manga Restoration](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Exploiting_Aliasing_for_Manga_Restoration_CVPR_2021_paper.pdf) | CVPR 2021 | [[webpage]](http://www.cse.cuhk.edu.hk/~ttwong/papers/mangarestore/mangarestore.html) [[code]](https://github.com/msxie92/MangaRestoration) |
| [Seamless Manga Inpainting with Semantics Awareness](https://www.cse.cuhk.edu.hk/~ttwong/papers/mangainpaint/mangainpaint.html) | SIGGRAPH 2021 | [[webpage]](https://www.cse.cuhk.edu.hk/~ttwong/papers/mangainpaint/mangainpaint.html) [[code]](https://github.com/msxie92/MangaInpainting) |


## 9. Sketch and Shading

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Learning to Shadow Hand-drawn Sketches](https://arxiv.org/abs/2002.11812) | CVPR 2020 | [[project]](https://cal.cs.umbc.edu/Papers/Zheng-2020-Shade/index.html) [[code]](https://github.com/qyzdao/ShadeSketch) |
| [SmartShadow: Artistic Shadow Drawing Tool for Line Drawings](https://lllyasviel.github.io/Style2PaintsResearch/) | ICCV 2021 |  |

## 10. Sketch Enhancement

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SketchMan: Learning to Create Professional Sketches](https://dl.acm.org/doi/abs/10.1145/3394171.3413720) | ACM MM 2020 | [[code]](https://github.com/LCXCUC/SketchMan2020) |


## 11. Sketch-Guided Object Localization 

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch-Guided Object Localization in Natural Images](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510528.pdf) | ECCV 2020 | [[project]](http://visual-computing.in/sketch-guided-object-localization/) [[code]](https://github.com/IISCAditayTripathi/SketchGuidedLocalization) |
| [Localizing Infinity-shaped fishes: Sketch-guided object localization in the wild](https://arxiv.org/abs/2109.11874) | arxiv 2109 | [[code]](https://github.com/priba/sgol_wild) |

## 12. Sketch-Guided Video Synthesis

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Deep Sketch-guided Cartoon Video Inbetweening](https://ieeexplore.ieee.org/abstract/document/9314221) | TVCG 2021 |  |


## 13. Sketch Segmentation and Perceptual Grouping

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
    <td rowspan="7"><strong>Stroke-level</strong></td>
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
    <td> arxiv 1901.03427 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/abs/2003.00678">SketchGCN: Semantic Sketch Segmentation with Graph Convolutional Networks</a> </td> 
    <td> arxiv 2003.00678 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="http://sweb.cityu.edu.hk/hongbofu/doc/SketchGNN_TOG21.pdf">SketchGNN: Semantic Sketch Segmentation with Graph Neural Networks</a> </td> 
    <td> TOG 2021 </td> 
    <td> <a href="https://github.com/sYeaLumin/SketchGNN">[code]</a> </td>
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


## 14. Sketch Representation Learning

- Stroke order importance/saliency, sketch abstraction

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [Pixelor: a competitive sketching AI agent. So you think you can sketch?](https://dl.acm.org/doi/pdf/10.1145/3414685.3417840) | SIGGRAPH Asia 2020 | [[Project]](http://sketchx.ai/pixelor) [[Code]](https://github.com/dasayan05/neuralsort-siggraph) | Vector/stroke-level | Sketch synthesis and recognition |
| [Goal-Driven Sequential Data Abstraction](http://openaccess.thecvf.com/content_ICCV_2019/papers/Muhammad_Goal-Driven_Sequential_Data_Abstraction_ICCV_2019_paper.pdf) | ICCV 2019 |  | Vector/stroke-level | Sketch recognition |
| [Learning Deep Sketch Abstraction](http://openaccess.thecvf.com/content_cvpr_2018/papers/Muhammad_Learning_Deep_Sketch_CVPR_2018_paper.pdf) | CVPR 2018 |  | Vector/stroke-level | FG-SBIR |

- Supervised Representation Learning

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [Im2Vec: Synthesizing Vector Graphics without Vector Supervision](https://arxiv.org/abs/2102.02798) | CVPR 2021 | [[Project]](http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/) [[code]](https://github.com/preddy5/Im2Vec) | SVG | Vector Graphics reconstruction and interpolation |
| [DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation](https://arxiv.org/abs/2007.11301) | NeurIPS 2020 | [[Code]](https://github.com/alexandre01/deepsvg)  [[Project]](https://blog.alexandrecarlier.com/deepsvg/) | SVG | Vector Graphics Animation, reconstruction and interpolation |
| [CoSE: Compositional Stroke Embeddings](https://papers.nips.cc/paper/2020/file/723e8f97fde15f7a8d5ff8d558ea3f16-Paper.pdf) | NeurIPS 2020 | [[Code]](https://github.com/eth-ait/cose) | Vector/stroke-level | Auto-completing diagrams |
| [Sketchformer: Transformer-based Representation for Sketched Structure](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ribeiro_Sketchformer_Transformer-Based_Representation_for_Sketched_Structure_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/leosampaio/sketchformer) | Vector/stroke-level | Sketch classification, SBIR, reconstruction and interpolation |
| [SketchHealer: A Graph-to-Sequence Network for Recreating Partial Human Sketches](https://core.ac.uk/download/pdf/334949144.pdf) | BMVC 2020 | [[code]](https://github.com/sgybupt/SketchHealer) | Vector/stroke-level | Sketch recognition, retrieval, completion and analogy |
| [A Learned Representation for Scalable Vector Graphics](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lopes_A_Learned_Representation_for_Scalable_Vector_Graphics_ICCV_2019_paper.pdf) | ICCV 2019 | [[code]](https://github.com/magenta/magenta/tree/master/magenta/models/svg_vae) | SVG | Font design |
| [A Neural Representation of Sketch Drawings (Sketch-RNN)](https://openreview.net/pdf?id=Hy6GHpkCW) | ICLR 2018 | [[code]](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn) | Vector/stroke-level | Reconstruction and interpolation |


- Self-supervised Representation Learning

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [Vectorization and Rasterization: Self-Supervised Learning for Sketch and Handwriting](https://arxiv.org/abs/2103.13716) | CVPR 2021 | [[Code]](https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch) | Vector/stroke-level | Recognition |
| [Deep Self-Supervised Representation Learning for Free-Hand Sketch](https://arxiv.org/abs/2002.00867) | TCSVT 2020 | [[Code]](https://github.com/zzz1515151/self-supervised_learning_sketch) | Vector/stroke-level | Retrieval and recognition |
| [Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt](https://arxiv.org/abs/2005.09159) | CVPR 2020 |  | Vector/stroke-level | Sketch recognition, retrieval, and gestalt |


- Pixel-level description and correspondence

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [SketchDesc: Learning Local Sketch Descriptors for Multi-view Correspondence](http://sweb.cityu.edu.hk/hongbofu/doc/SketchDesc_TCSVT2020.pdf) | TCSVT 2020 |  | Pixel-level | Semantic correspondence among multi-view sketches |
| [SketchTransfer: A Challenging New Task for Exploring Detail-Invariance and the Abstractions Learned by Deep Networks](https://arxiv.org/pdf/1912.11570.pdf) | WACV 2020 |  | Pixel-level | Domain transfer learning |


- Other kinds of sketch representation

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [SketchLattice: Latticed Representation for Sketch Manipulation](https://arxiv.org/abs/2108.11636) | ICCV 2021 |  | Lattice graph | Sketch healing and image-to-sketch synthesis |


## 15. Sketch Animation/Inbetweening

- Inbetweening

| Paper | Source | Representation | Code/Project Link |
| --- | --- | :--: | --- |
| [BetweenIT: An Interactive Tool for Tight Inbetweening](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2009.01630.x) | CGF 2010 | stroke |  |
| [Context-Aware Computer Aided Inbetweening](https://ieeexplore.ieee.org/abstract/document/7831370) | TVCG 2018 | stroke |  |
| [FTP-SC: Fuzzy Topology Preserving Stroke Correspondence](https://dcgi.fel.cvut.cz/home/sykorad/Yang18-SCA.pdf) | SCA 2018 | stroke | [[webpage]](https://dcgi.fel.cvut.cz/home/sykorad/FTP-SC.html) [[video]](https://youtu.be/3oZfCAkYJQk) |
| [Cacani: 2d animation and inbetween software](https://cacani.sg/?v=1c2903397d88) | / | stroke | [[software]](https://cacani.sg/?v=1c2903397d88) |
| [Optical Flow Based Line Drawing Frame Interpolation Using Distance Transform to Support Inbetweenings](https://ieeexplore.ieee.org/abstract/document/8803506) | ICIP 2019 | raster |  |

- Animation

| Paper | Source | Representation | Code/Project Link  |
| --- | --- | :--: | --- |
| [Autocomplete Hand-drawn Animations](http://junxnui.github.io/research/siga15_autocomplete_handdrawn_animations.pdf) | SIGGRAPH Asia 2015 | stroke | [[webpage]](https://iis-lab.org/research/autocomplete-animations/) [[video]](https://youtu.be/w0YmWiy6sA4) |
| [Live Sketch: Video-driven Dynamic Deformation of Static Drawings](http://sweb.cityu.edu.hk/hongbofu/doc/livesketch_CHI2018.pdf) | CHI 2018 | vector | [[video]](https://youtu.be/6DjQR5k286E) |


## 16. Sketch and AR/VR

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SweepCanvas: Sketch-based 3D Prototyping on an RGB-D Image](http://sweb.cityu.edu.hk/hongbofu/doc/sweep_canvas_UIST2017.pdf) | UIST 2017 | [[video]](https://youtu.be/Xnp3_eMYXj0) |
| [Model-Guided 3D Sketching](http://sweb.cityu.edu.hk/hongbofu/doc/model-guided_3D_sketching_TVCG.pdf) | TVCG 2018 | [[video]](https://youtu.be/STredKjB_Bk) |
| [Mobi3DSketch: 3D Sketching in Mobile AR](http://sweb.cityu.edu.hk/hongbofu/doc/mobi3Dsketch_CHI2019.pdf) | CHI 2019 | [[video]](https://youtu.be/JdP0nkeMEog) |
| [Interactive Liquid Splash Modeling by User Sketches](https://dl.acm.org/doi/abs/10.1145/3414685.3417832) | SIGGRAPH Asia 2020 | [[video]](https://youtu.be/HXAxNrfk_w0) |
| [3D Curve Creation on and around Physical Objects with Mobile AR](http://sweb.cityu.edu.hk/hongbofu/doc/3D_Curve_Creation_Mobile_AR_TVCG.pdf) | TVCG 2021 | [[video]](https://youtu.be/zyh4pEvK7j8) |
| [HandPainter - 3D Sketching in VR with Hand-based Physical Proxy](https://dl.acm.org/doi/abs/10.1145/3411764.3445302) | CHI 2021 | [[video]](https://youtu.be/x5VAU-471P8) |
