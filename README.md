# Awesome-Sketch-Based-Applications

A collection of sketch based applications.

> Feel free to create a PR or an issue.


**Outline**

- [0. Survey](#0-survey)
- [1. Sketch Based Image Synthesis](#1-sketch-based-image-synthesis)
  - [1.1 Automatic Synthesis](#11-automatic-synthesis)
  - [1.2 Style-conditioned](#12-style-conditioned)
  - [1.3 Text-conditioned](#13-text-conditioned)
- [2. Sketch Based Image Editing](#2-sketch-based-image-editing)
- [3. Sketch Based Image Retrieval (SBIR)](#3-sketch-based-image-retrieval-sbir)
- [4. Sketch Based 3D Shape Retrieval](#4-sketch-based-3d-shape-retrieval)
- [5. Sketch Based 3D Shape Modeling](#5-sketch-based-3d-shape-modeling)
- [6. Sketch Based Garment Design](#6-sketch-based-garment-design)
- [7. Sketch Completion](#7-sketch-completion)
- [8. Sketch Restoration, Retargeting and Inpainting](#8-sketch-restoration-retargeting-and-inpainting)
- [9. Sketch and Shading](#9-sketch-and-shading)
- [10. Sketch Enhancement](#10-sketch-enhancement)
- [11. Sketch-Guided Object Localization](#11-sketch-guided-object-localization)
- [12. Sketch-Guided Video Synthesis](#12-sketch-guided-video-synthesis)
- [13. Sketch Recognition](#13-sketch-recognition)
- [14. Sketch Segmentation and Perceptual Grouping](#14-sketch-segmentation-and-perceptual-grouping)
- [15. Sketch Representation Learning](#15-sketch-representation-learning)
- [16. Sketch and Visual Correspondence](#16-sketch-and-visual-correspondence)
- [17. Sketch Animation/Inbetweening](#17-sketch-animationinbetweening)
- [18. Sketch and AR/VR](#18-sketch-and-arvr)
- [19. Sketch Based Incremental Learning](#19-sketch-based-incremental-learning)
- [20. Sketch Quality Measurement](#20-sketch-quality-measurement)
- [21. Cloud Augmentation with Sketches](#21-cloud-augmentation-with-sketches)
- [22. Sketch and Re-identification](#22-sketch-and-re-identification)
- [23. Sketch-based Salient Object Detection](#23-sketch-based-salient-object-detection)

---

## 0. Survey

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Deep Learning for Free-Hand Sketch: A Survey](https://ieeexplore.ieee.org/abstract/document/9706366) | TPAMI 2022 | [[code]](https://github.com/PengBoXiangShang/torchsketch) |

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
| [Sketch Your Own GAN](https://arxiv.org/abs/2108.02774) | ICCV 2021 | [[code]](https://github.com/peterwang512/GANSketching) [[webpage]](https://peterwang512.github.io/GANSketching/) |
| [Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis](https://arxiv.org/abs/2104.05703) | WACV 2022 | [[code]](https://github.com/Mukosame/AODA) |
| [Customizing GAN Using Few-Shot Sketches](https://dl.acm.org/doi/abs/10.1145/3503161.3548415) | ACM MM 2022 |  |
| [DiffSketching: Sketch Control Image Synthesis with Diffusion Models](https://bmvc2022.mpi-inf.mpg.de/0067.pdf) | BMVC 2022 | [[code]](https://github.com/XDUWQ/DiffSketching) | 
| [MaskSketch: Unpaired Structure-guided Masked Image Generation](https://arxiv.org/abs/2302.05496) | CVPR 2023 | [[project]](https://masksketch.github.io/) [[code]](https://github.com/google-research/masksketch)| 
| [Picture that Sketch: Photorealistic Image Generation from Abstract Sketches](https://arxiv.org/abs/2303.11162) | CVPR 2023 | [[project]](https://subhadeepkoley.github.io/PictureThatSketch/) [[code]](https://github.com/subhadeepkoley/PictureThatSketch) | 


- Human Face / Portrait

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [LinesToFacePhoto: Face Photo Generation from Lines with Conditional Self-Attention Generative Adversarial Network](https://arxiv.org/pdf/1910.08914.pdf) | ACM MM 2019 |  |
| [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf) | SIGGRAPH 2020 | [[project]](http://geometrylearning.com/DeepFaceDrawing/) [[code]](https://github.com/IGLICT/DeepFaceDrawing-Jittor) |
| [DeepFacePencil: Creating Face Images from Freehand Sketches](https://arxiv.org/abs/2008.13343) | ACM MM 2020 | [[project]](https://liyuhangustc.github.io/Sketch2Face/) [[code]](https://github.com/LiYuhangUSTC/Sketch2Face) |
| [Controllable Sketch-to-Image Translation for Robust Face Synthesis](https://ieeexplore.ieee.org/abstract/document/9583954) | TIP 2021 |  |
| [DrawingInStyles: Portrait Image Generation and Editing with Spatially Conditioned StyleGAN](http://sweb.cityu.edu.hk/hongbofu/doc/DrawingInStyles_TVCG22.pdf) | TVCG 2022 |  |
| [DeepPortraitDrawing: Generating Human Body Images from Freehand Sketches](https://arxiv.org/abs/2205.02070) | C&G 2023 |  |
| [Semantics-Preserving Sketch Embedding for Face Generation](https://arxiv.org/abs/2211.13015) | TMM 2023 | [[project]](http://staff.ustc.edu.cn/~xjchen99/sketchFaceTMM/sketchFaceTmm.htm) [[code]](https://github.com/BinxinYang/Semantics-Preserving-Sketch-Embedding-for-Face-Generation) |
| [Parsing-Conditioned Anime Translation: A New Dataset and Method](https://dl.acm.org/doi/abs/10.1145/3585002) | TOG 2023 | [[code]](https://github.com/zsl2018/StyleAnime) |

- 3D image

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [3D-aware Conditional Image Synthesis](https://arxiv.org/abs/2302.08509) | CVPR 2023 | [[project]](https://www.cs.cmu.edu/~pix2pix3D/) [[code]](https://github.com/dunbar12138/pix2pix3D) |

### 1.2 Style-conditioned

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [TextureGAN: Controlling Deep Image Synthesis with Texture Patches](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_TextureGAN_Controlling_Deep_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/janesjanes/Pytorch-TextureGAN) |
| [Multimodal Unsupervised Image-to-Image Translation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/nvlabs/MUNIT) |
| [SketchPatch: Sketch Stylization via Seamless Patch-level Synthesis](https://arxiv.org/abs/2009.02216) | SIGGRAPH Asia 2020 |  |
| [Self-Supervised Sketch-to-Image Synthesis](https://arxiv.org/abs/2012.09290) | AAAI 2021 | [[code]](https://github.com/odegeasslbc/Self-Supervised-Sketch-to-Image-Synthesis-PyTorch) |
| [CoGS: Controllable Generation and Search from Sketch and Style](https://arxiv.org/abs/2203.09554) | ECCV 2022 |  |
| [Adaptively-Realistic Image Generation from Stroke and Sketch with Diffusion Model](https://arxiv.org/abs/2208.12675) | WACV 2023 | [[project]](https://cyj407.github.io/DiSS/) [[code]](https://github.com/cyj407/DiSS) |
| [DemoCaricature: Democratising Caricature Generation with a Rough Sketch](https://arxiv.org/abs/2312.04364) | CVPR 2024 | [[project]](https://democaricature.github.io/) |

### 1.3 Text-conditioned

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch-Guided Text-to-Image Diffusion Models](https://arxiv.org/abs/2211.13752) | SIGGRAPH 2023 | [[project]](https://sketch-guided-diffusion.github.io/) |
| [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) | ICCV 2023 | [[code]](https://github.com/lllyasviel/ControlNet) |
| [T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453) | arxiv 23.02 | [[code]](https://github.com/TencentARC/T2I-Adapter) |
| [UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild](https://arxiv.org/abs/2305.11147) | NeurIPS 2023 | [[code]](https://github.com/salesforce/UniControl) [[project]](https://canqin001.github.io/UniControl-Page/) |
| [Stable Doodle (ClipDrop)](https://stability.ai/blog/clipdrop-launches-stable-doodle?continueFlag=6b17b7a626134700c01cc2e1afe3158a) |  | [[demo]](https://clipdrop.co/stable-doodle) |
| [CustomSketching: Sketch Concept Extraction for Sketch-based Image Synthesis and Editing](https://arxiv.org/abs/2402.17624) | arxiv 24.02 |  |
| [Block and Detail: Scaffolding Sketch-to-Image Generation](https://arxiv.org/abs/2402.18116) | arxiv 24.02 |  |
| [One-Step Image Translation with Text-to-Image Models](https://arxiv.org/abs/2403.12036) | arxiv 24.03 | [[code]](https://github.com/GaParmar/img2img-turbo) |
| [It's All About Your Sketch: Democratising Sketch Control in Diffusion Models](https://arxiv.org/abs/2403.07234) | CVPR 2024 | [[code]](https://github.com/subhadeepkoley/StableSketching) |

## 2. Sketch Based Image Editing

- Arbitrary Image

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589) (DeepFill v2) | ICCV 2019 | [[project]](http://jiahuiyu.com/deepfill) [[code]](https://github.com/JiahuiYu/generative_inpainting) |
| [Fashion Editing with Multi-scale Attention Normalization](https://arxiv.org/pdf/1906.00884.pdf) | 1906.00884 |  |
| [DeFLOCNet: Deep Image Editing via Flexible Low-level Controls](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_DeFLOCNet_Deep_Image_Editing_via_Flexible_Low-Level_Controls_CVPR_2021_paper.pdf) | CVPR 2021 | [[code]](https://github.com/KumapowerLIU/DeFLOCNet) |
| [DeepSIM: Image Shape Manipulation from a Single Augmented Training Sample](https://arxiv.org/abs/2109.06151) | ICCV 2021 | [[code]](https://github.com/eliahuhorwitz/DeepSIM) [[project]](http://www.vision.huji.ac.il/deepsim/) |
| [SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches](https://arxiv.org/abs/2111.15078) | CVPR 2022 | [[code]](https://github.com/zengxianyu/sketchedit) [[project]](https://zengxianyu.github.io/sketchedit/) |
| [Draw2Edit: Mask-Free Sketch-Guided Image Manipulation](https://dl.acm.org/doi/abs/10.1145/3581783.3612398) | ACM MM 2023 | [[code]](https://github.com/YiwenXu/Draw2Edit)  |

- Human Face / Portrait / Hair

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [FaceShop: Deep Sketch-based Face Image Editing](https://arxiv.org/pdf/1804.08972.pdf) | SIGGRAPH 2018 | [[project]](https://home.inf.unibe.ch/~porteni/projects/faceshop/) |
| [CaricatureShop: Personalized and Photorealistic Caricature Sketching](https://ieeexplore.ieee.org/document/8580421) | TVCG 2018 |  |
| [Sparse, Smart Contours to Represent and Edit Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dekel_Sparse_Smart_Contours_CVPR_2018_paper.pdf) | CVPR 2018 | [[project]](https://contour2im.github.io/) |
| [Example-Guided Style-Consistent Image Synthesis from Semantic Labeling](https://arxiv.org/pdf/1906.01314) | CVPR 2019 | [[code]](https://github.com/cxjyxxme/pix2pixSC) |
| [SC-FEGAN: Face Editing Generative Adversarial Network With User's Sketch and Color](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jo_SC-FEGAN_Face_Editing_Generative_Adversarial_Network_With_Users_Sketch_and_ICCV_2019_paper.pdf) | ICCV 2019 | [[code]](https://github.com/run-youngjoo/SC-FEGAN) |
| [Intuitive, Interactive Beard and Hair Synthesis with Generative Models](https://arxiv.org/abs/2004.06848) | CVPR 2020 |  |
| [Deep Plastic Surgery: Robust and Controllable Image Editing with Human-Drawn Sketches](https://arxiv.org/abs/2001.02890) | ECCV 2020 | [[code]](https://github.com/VITA-Group/DeepPS) [[project]](https://williamyang1991.github.io/projects/ECCV2020/) |
| [DeepFaceEditing: Deep Face Generation and Editing with Disentangled Geometry and Appearance Control](http://www.geometrylearning.com/DeepFaceEditing/) | SIGGRAPH 2021 | [[code]](https://github.com/IGLICT/DeepFaceEditing-Jittor) [[project]](http://www.geometrylearning.com/DeepFaceEditing/) |
| [SketchHairSalon: Deep Sketch-based Hair Image Synthesis](https://arxiv.org/abs/2109.07874) | SIGGRAPH Asia 2021 | [[project]](https://chufengxiao.github.io/SketchHairSalon/) |
| [Paint2Pix: Interactive Painting based Progressive Image Synthesis and Editing](https://arxiv.org/abs/2208.08092) | ECCV 2022 | [[code]](https://github.com/1jsingh/paint2pix) [[project]](https://1jsingh.github.io/paint2pix) |
| [NeRFFaceEditing: Disentangled Face Editing in Neural Radiance Fields](https://dl.acm.org/doi/abs/10.1145/3550469.3555377) | SIGGRAPH Asia 2022 | [[project]](http://geometrylearning.com/NeRFFaceEditing/) |
| [SketchFaceNeRF: Sketch-based Facial Generation and Editing in Neural Radiance Fields](https://orca.cardiff.ac.uk/id/eprint/159468/1/NeRFFaceSketch_SIG23.pdf) | SIGGRAPH 2023 |  |
| [DemoCaricature: Democratising Caricature Generation with a Rough Sketch](https://arxiv.org/abs/2312.04364) | arxiv 23.12 | [[project]](https://democaricature.github.io/) [[code]](https://github.com/ChenDarYen/DemoCaricature/) |

- Anime Editing

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [AniFaceDrawing: Anime Portrait Exploration during Your Sketching](https://arxiv.org/abs/2306.07476) | SIGGRAPH 2023 | [[project]](http://www.jaist.ac.jp/~xie/AniFaceDrawing.html) |

- Video Editing

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [DeepFaceVideoEditing: Sketch-based Deep Editing of Face Videos](http://geometrylearning.com/DeepFaceVideoEditing/) | SIGGRAPH 2022 | [[project]](http://geometrylearning.com/DeepFaceVideoEditing/) |


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
| [Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Semantic-Aware_Knowledge_Preservation_for_Zero-Shot_Sketch-Based_Image_Retrieval_ICCV_2019_paper.pdf) | ICCV 2019 |  |
| [TC-Net for iSBIR: Triplet Classification Network for Instance-level Sketch Based Image Retrieval](http://www.eecs.qmul.ac.uk/~sgg/papers/LinEtAl_ACM_MM2019.pdf) | ACM MM 2019 |  |
| [Sketch-Based Image Retrieval With Multi-Clustering Re-Ranking](https://ieeexplore.ieee.org/abstract/document/8933028) | TCSVT 2019 |  |
| [Semi-Heterogeneous Three-Way Joint Embedding Network for Sketch-Based Image Retrieval](https://ieeexplore.ieee.org/abstract/document/8809264) | TCSVT 2019 |  |
| [Zero-Shot Sketch-Based Image Retrieval via Graph Convolution Network](https://ojs.aaai.org/index.php/AAAI/article/view/6993) | AAAI 2020 |  |
| [Sketch Less for More: On-the-Fly Fine-Grained Sketch Based Image Retrieval](https://arxiv.org/abs/2002.10310) | CVPR 2020 | [[code]](https://github.com/AyanKumarBhunia/on-the-fly-FGSBIR) |
| [Solving Mixed-modal Jigsaw Puzzle for Fine-Grained Sketch-Based Image Retrieval](http://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Solving_Mixed-Modal_Jigsaw_Puzzle_for_Fine-Grained_Sketch-Based_Image_Retrieval_CVPR_2020_paper.pdf) | CVPR 2020 |  |
| [Fine-Grained Instance-Level Sketch-Based Image Retrieval](https://link.springer.com/article/10.1007/s11263-020-01382-3) | IJCV 2020 |  |
| [StyleMeUp: Towards Style-Agnostic Sketch-Based Image Retrieval](https://arxiv.org/abs/2103.15706) | CVPR 2021 |  |
| [More Photos are All You Need: Semi-Supervised Learning for Fine-Grained Sketch-Based Image Retrieval](https://arxiv.org/abs/2103.13990) | CVPR 2021 | [[code]](https://github.com/AyanKumarBhunia/semisupervised-FGSBIR) |
| [DLA-Net for FG-SBIR: Dynamic Local Aligned Network for Fine-Grained Sketch-Based Image Retrieval](https://dl.acm.org/doi/abs/10.1145/3474085.3475705) | ACM MM 2021 |  |
| [Domain-Aware SE Network for Sketch-based Image Retrieval with Multiplicative Euclidean Margin Softmax](https://dl.acm.org/doi/abs/10.1145/3474085.3475499) | ACM MM 2021 | [[code]](https://github.com/Ben-Louis/SBIR-DASE-MEMS) |
| [Relationship-Preserving Knowledge Distillation for Zero-Shot Sketch Based Image Retrieval](https://dl.acm.org/doi/abs/10.1145/3474085.3475676) | ACM MM 2021 |  |
| [Transferable Coupled Network for Zero-Shot Sketch-Based Image Retrieval](https://ieeexplore.ieee.org/abstract/document/9591307) | TPAMI 2021 | [[project]](https://haowang1992.github.io/publication/TCN) |
| [TVT: Three-Way Vision Transformer through Multi-Modal Hypersphere Learning for Zero-Shot Sketch-Based Image Retrieval](https://www.aaai.org/AAAI22Papers/AAAI-8379.TianJ.pdf) | AAAI 2022 |  |
| [Sketching without Worrying: Noise-Tolerant Sketch-Based Image Retrieval](https://arxiv.org/abs/2203.14817) | CVPR 2022 | [[code]](https://github.com/ayankumarbhunia/stroke_subset_selector-for-fgsbir) |
| [Sketch3T: Test-time Training for Zero-Shot SBIR](https://arxiv.org/abs/2203.14691) | CVPR 2022 |  |
| [Augmented Multi-Modality Fusion for Generalized Zero-Shot Sketch-based Visual Retrieval](https://ieeexplore.ieee.org/abstract/document/9775617) | TIP 2022 | [[code]](https://github.com/scottjingtt/AMF_GZS_SBIR) |
| [Adaptive Fine-Grained Sketch-Based Image Retrieval](https://arxiv.org/abs/2207.01723) | ECCV 2022 | [[code]](https://github.com/AyanKumarBhunia/Adaptive-FGSBIR) |
| [Conditional Stroke Recovery for Fine-Grained Sketch-Based Image Retrieval](https://github.com/1069066484/CSR-ECCV2022) | ECCV 2022 | [[code]](https://github.com/1069066484/CSR-ECCV2022) |
| [A Sketch Is Worth a Thousand Words: Image Retrieval with Text and Sketch](https://patsorn.me/projects/tsbir/paper.pdf) | ECCV 2022 | [[code]](https://github.com/janesjanes/tsbir) [[project]](https://patsorn.me/projects/tsbir/) |
| [Multi-Level Region Matching for Fine-Grained Sketch-Based Image Retrieval](https://www.jiangtongli.me/publication/mlmr/mlmr.pdf) | ACM MM 2022 | [[code]](https://github.com/1069066484/MLRM-ACMMM2022) |
| [Prototype-based Selective Knowledge Distillation for Zero-Shot Sketch Based Image Retrieval](https://dl.acm.org/doi/abs/10.1145/3503161.3548382) | ACM MM 2022 |  |
| [DLI-Net: Dual Local Interaction Network for Fine-Grained Sketch-Based Image Retrieval](https://ieeexplore.ieee.org/abstract/document/9766165) | TCSVT 2022 | [[code]](https://github.com/xjq1998/DLI-Net) |
| [Data-Free Sketch-Based Image Retrieval](https://openaccess.thecvf.com/content/CVPR2023/papers/Chaudhuri_Data-Free_Sketch-Based_Image_Retrieval_CVPR_2023_paper.pdf) | CVPR 2023 | [[code]](https://github.com/abhrac/data-free-sbir) |
| [Zero-Shot Everything Sketch-Based Image Retrieval, and in Explainable Style](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Zero-Shot_Everything_Sketch-Based_Image_Retrieval_and_in_Explainable_Style_CVPR_2023_paper.pdf) | CVPR 2023 | [[code]](https://github.com/buptLinfy/ZSE-SBIR) |
| [Exploiting Unlabelled Photos for Stronger Fine-Grained SBIR](https://arxiv.org/abs/2303.13779) | CVPR 2023 | [[project]](https://aneeshan95.github.io/Sketch_PVT/) |
| [CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not](https://arxiv.org/abs/2303.13440) | CVPR 2023 | [[project]](https://aneeshan95.github.io/Sketch_LVM/) |
| [Cross-Domain Alignment for Zero-Shot Sketch-Based Image Retrieval](https://ieeexplore.ieee.org/abstract/document/10098211) | TCSVT 2023 |  |
| [Semi-transductive Learning for Generalized Zero-Shot Sketch-Based Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/25931) | AAAI 2023 |  |
| [Text-to-Image Diffusion Models are Great Sketch-Photo Matchmakers](https://arxiv.org/abs/2403.07214) | CVPR 2024 | [[project]](https://subhadeepkoley.github.io/DiffusionZSSBIR/) |
| [You'll Never Walk Alone: A Sketch and Text Duet for Fine-Grained Image Retrieval](https://arxiv.org/abs/2403.07222) | CVPR 2024 | [[project]](https://subhadeepkoley.github.io/SBCIR/) |
| [How to Handle Sketch-Abstraction in Sketch-Based Image Retrieval?](https://arxiv.org/abs/2403.07203) | CVPR 2024 | [[project]](https://subhadeepkoley.github.io/AbstractAway/) |
| [Asymmetric Mutual Alignment for Unsupervised Zero-Shot Sketch-Based Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/29588) | AAAI 2024 |  |


- Scene-level

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SceneSketcher: Fine-Grained Image Retrieval with Scene Sketches](http://orca.cf.ac.uk/133561/1/SceneSketcherECCV2020.pdf) | ECCV 2020 |  |
| [Partially Does It: Towards Scene-Level FG-SBIR with Partial Input](https://arxiv.org/abs/2203.14804) | CVPR 2022 |  |
| [SceneSketcher-v2: Fine-Grained Scene-Level Sketch-Based Image Retrieval using Adaptive GCNs](https://ieeexplore.ieee.org/abstract/document/9779565) | TIP 2022 |  |
| [FS-COCO: Towards Understanding of Freehand Sketches of Common Objects in Context](https://ieeexplore.ieee.org/abstract/document/9779565) | ECCV 2022 | [[code]](https://github.com/pinakinathc/fscoco) [[Dataset]](https://fscoco.github.io/) |
| [Scene-Level Sketch-Based Image Retrieval with Minimal Pairwise Supervision](https://ojs.aaai.org/index.php/AAAI/article/view/25141) | AAAI 2023 |  |

- Video Retrieval

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Fine-Grained Instance-Level Sketch-Based Video Retrieval](https://ieeexplore.ieee.org/abstract/document/9161000) | TCSVT 2020 |  |
| [Fine-Grained Video Retrieval with Scene Sketches](https://ieeexplore.ieee.org/abstract/document/10136606/) | TIP 2023 | [[project]](https://iscas-mmsketch.github.io/FG-SL-SBVR/) |


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
| [Uncertainty Learning for Noise Resistant Sketch-Based 3D Shape Retrieval](https://ieeexplore.ieee.org/abstract/document/9573502) | TIP 2021 |  |
| [Towards Fine-Grained Sketch-Based 3D Shape Retrieval](https://ieeexplore.ieee.org/abstract/document/9573376) | TIP 2021 |  |
| [Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval](https://arxiv.org/abs/2202.11948) | AAAI 2022 |  |
| [Retrieval-Specific View Learning for Sketch-to-Shape Retrieval](https://ieeexplore.ieee.org/abstract/document/10155453) | TMM 2023 |  |
| [Doodle to Object: Practical Zero-Shot Sketch-Based 3D Shape Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/25344) | AAAI 2023 | [[code]](https://github.com/yigohw/doodle2object) |
| [Democratising 2D Sketch to 3D Shape Retrieval Through Pivoting](https://openaccess.thecvf.com/content/ICCV2023/papers/Chowdhury_Democratising_2D_Sketch_to_3D_Shape_Retrieval_Through_Pivoting_ICCV_2023_paper.pdf) | ICCV 2023 |  |


## 5. Sketch Based 3D Shape Modeling

- Free-hand sketch

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks](https://people.cs.umass.edu/~zlun/papers/SketchModeling/SketchModeling.pdf) | 3DV 2017 | [[code]](https://github.com/happylun/SketchModeling) [[project]](https://people.cs.umass.edu/~zlun/SketchModeling/) |
| [Shape Synthesis from Sketches via Procedural Models and Convolutional Networks](https://people.cs.umass.edu/~kalo/papers/shapepmconvnet/shapepmconvnet.pdf) | TVCG 2017 |  |
| [DeepSketch2Face: A Deep Learning Based Sketching System for 3D Face and Caricature Modeling](https://i.cs.hku.hk/~xghan/papers/deepske2face.pdf) | SIGGRAPH 2017 | [[project]](https://i.cs.hku.hk/~xghan/Projects/ske2face.htm) [[code]](https://github.com/changgyhub/deepsketch2face) |
| [BendSketch: Modeling Freeform Surfaces Through 2D Sketching](http://haopan.github.io/papers/bendsketch.pdf) | SIGGRAPH 2017 | [[project]](http://haopan.github.io/bendsketch.html) |
| [Robust Flow-Guided Neural Prediction for Sketch-Based Freeform Surface Modeling](http://haopan.github.io/papers/SketchCNN.pdf) | SIGGRAPH Asia 2018 | [[code]](https://github.com/Enigma-li/SketchCNN) [[project]](http://haopan.github.io/sketchCNN.html) |
| [DeepSketchHair: Deep Sketch-based 3D Hair Modeling](https://arxiv.org/abs/1908.07198) | TVCG 2019 |  |
| [Interactive Liquid Splash Modeling by User Sketches](https://dl.acm.org/doi/abs/10.1145/3414685.3417832) | SIGGRAPH Asia 2020 |  |
| [Monster Mash: A Single-View Approach to Casual 3D Modeling and Animation](https://dcgi.fel.cvut.cz/home/sykorad/Dvoroznak20-SA.pdf) | SIGGRAPH Asia 2020 | [[project]](https://dcgi.fel.cvut.cz/home/sykorad/monster_mash) [[code]](https://github.com/google/monster-mash) [[demo]](http://monstermash.zone/) |
| [SAniHead: Sketching Animal-like 3D Character Heads Using a View-surface Collaborative Mesh Generative Network](http://sweb.cityu.edu.hk/hongbofu/doc/2020TVCG_SAniHead.pdf) | TVCG 2020 |  |
| [Towards Practical Sketch-Based 3D Shape Generation: The Role of Professional Sketches](https://ieeexplore.ieee.org/abstract/document/9272370) | TCSVT 2020 |  |
| [Sketch2Model: View-Aware 3D Modeling from Single Free-Hand Sketches](https://arxiv.org/abs/2105.06663) | CVPR 2021 |  |
| [Sketch2Mesh: Reconstructing and Editing 3D Shapes from Sketches](https://arxiv.org/abs/2104.00482v1) | ICCV 2021 |  |
| [Real-time Skeletonization for Sketch-based Modeling](https://arxiv.org/abs/2110.05805) | SMI 2021 | [[code]](https://github.com/jingma-git/RealSkel) |
| [Sketch2Pose: Estimating a 3D Character Pose from a Bitmap Sketch](http://www-labs.iro.umontreal.ca/~bmpix/sketch2pose/sketch2pose.pdf) | SIGGRAPH 2022 | [[project]](http://www-ens.iro.umontreal.ca/~brodtkir/projects/sketch2pose/) [[code]](https://github.com/kbrodt/sketch2pose) |
| [Sketch2PQ: Freeform Planar Quadrilateral Mesh Design via a Single Sketch](https://arxiv.org/abs/2201.09367) | TVCG 2022 |  |
| [Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation](https://arxiv.org/abs/2207.09019) | ECCV 2022 | [[code]](https://github.com/gerwang/facial-detail-manipulation) |
| [SketchSampler: Sketch-based 3D Reconstruction via View-dependent Depth Sampling](https://arxiv.org/abs/2208.06880) | ECCV 2022 | [[code]](https://github.com/cjeen/sketchsampler) |
| [Deep Reconstruction of 3D Smoke Densities from Artist Sketches](http://www.byungsoo.me/project/sketch2density/paper.pdf) | EG 2022 | [[project]](http://www.byungsoo.me/project/sketch2density/) [[code]](https://github.com/byungsook/sketch2fluid) |
|[A Diffusion-ReFinement Model for Sketch-to-Point Modeling](https://openaccess.thecvf.com/content/ACCV2022/papers/Kong_A_Diffusion-ReFinement_Model_for_Sketch-to-Point_Modeling_ACCV_2022_paper.pdf)| ACCV 2022 | [[code]](https://github.com/Walterkd/diffusion-refine-sketch2point) |
|[RaBit: Parametric Modeling of 3D Biped Cartoon Characters with a Topological-consistent Dataset](https://arxiv.org/abs/2303.12564)| CVPR 2023 | [[project]](https://gaplab.cuhk.edu.cn/projects/RaBit/) |
|[SketchMetaFace: A Learning-based Sketching Interface for High-fidelity 3D Character Face Modeling](https://arxiv.org/abs/2307.00804)| TVCG 2023 | [[project]](https://zhongjinluo.github.io/SketchMetaFace/) [[code]](https://github.com/zhongjinluo/SketchMetaFace) |
|[Reality3DSketch: Rapid 3D Modeling of Objects from Single Freehand Sketches](https://arxiv.org/abs/2310.18148)| TMM 2023 |  |
|[Doodle Your 3D: From Abstract Freehand Sketches to Precise 3D Shapes](https://arxiv.org/abs/2312.04043)| CVPR 2024 | [[project]](https://hmrishavbandy.github.io/doodle23d/) |


- NeRF

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch2NeRF: Multi-view Sketch-guided Text-to-3D Generation](https://arxiv.org/abs/2401.14257) | arxiv 24.01 |  |
| [SKED: Sketch-guided Text-based 3D Editing](https://arxiv.org/abs/2303.10735) | ICCV 2023 | [[project]](https://sked-paper.github.io/) |

- CAD sketch

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Lifting Freehand Concept Sketches into 3D](https://repo-sam.inria.fr/d3/Lift3D/Gryaditskaya_SigAsia20_Lifting%20_Freehand_Concept_Sketches_into_3D.pdf) | SIGGRAPH Asia 2020 | [[project]](https://ns.inria.fr/d3/Lift3D/) [[code]](https://github.com/ygryadit/LiftConceptSketches3D) |
| [Sketch2CAD: Sequential CAD Modeling by Sketching in Context](http://geometry.cs.ucl.ac.uk/projects/2020/sketch2cad/paper_docs/Sketch2CAD_SIGA_2020.pdf) | SIGGRAPH Asia 2020 | [[project]](http://geometry.cs.ucl.ac.uk/projects/2020/sketch2cad/) [[code]](https://github.com/Enigma-li/Sketch2CAD) |
| [Free2CAD: Parsing Freehand Drawings into CAD Commands](https://enigma-li.github.io/projects/free2cad/Free2CAD_SIG_2022.pdf) | SIGGRAPH 2022 | [[project]](http://geometry.cs.ucl.ac.uk/projects/2022/free2cad/) [[code]](https://github.com/Enigma-li/Free2CAD) |
| [Symmetry-driven 3D Reconstruction From Concept Sketches](https://repo-sam.inria.fr/d3/SymmetrySketch/symmetry_sketch.pdf) | SIGGRAPH 2022 | [[project]](https://ns.inria.fr/d3/SymmetrySketch/) |
| [Piecewise-smooth Surface Fitting Onto Unstructured 3D Sketches](http://www-sop.inria.fr/reves/Basilic/2022/YABSB22/surfacing_sketches.pdf) | SIGGRAPH 2022 | [[project]](https://em-yu.github.io/research/surfacing_3d_sketches/) |
| [Reconstruction of Machine-Made Shapes from Bitmap Sketches](https://dl.acm.org/doi/abs/10.1145/3618361) | SIGGRAPH Asia 2023 | [[project]](https://puhachov.xyz/publications/machine-made-sketch-reconstruction/) |
| [CAD-SIGNet: CAD Language Inference from Point Clouds using Layer-wise Sketch Instance Guided Attention](https://arxiv.org/abs/2402.17678) | CVPR 2024 |  |


## 6. Sketch Based Garment Design

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketching Folds: Developable Surfaces from Non-Planar Silhouettes](http://imagecomputing.net/damien.rohmer/publications/2015_tog_sketching_folds/article/JungHRBBC_TOG_2015_sketching_folds.pdf) | TOG 2015 |  [[project]](http://imagecomputing.net/damien.rohmer/publications/2015_tog_sketching_folds/index.html) |
| [FoldSketch: Enriching Garments with Physically Reproducible Folds](http://www.cs.ubc.ca/labs/imager/tr/2018/FoldSketch/doc/FoldSketch.pdf) | SIGGRAPH 2018 |  [[project]](http://www.cs.ubc.ca/labs/imager/tr/2018/FoldSketch/) |
| [Learning a Shared Shape Space for Multimodal Garment Design](https://arxiv.org/abs/1806.11335) | SIGGRAPH Asia 2018 |  [[project]](http://geometry.cs.ucl.ac.uk/projects/2018/garment_design/) |
| [Garment Ideation: Iterative view-aware sketch-based garment modeling](https://github.com/pinakinathc/multiviewsketch-garment) | 3DV 2022 | [[code]](https://github.com/pinakinathc/multiviewsketch-garment) |
| [Controllable Visual-Tactile Synthesis](https://arxiv.org/pdf/2305.03051.pdf) | ICCV 2023 | [[project]](https://visual-tactile-synthesis.github.io/) [[code]](https://github.com/RuihanGao/visual-tactile-synthesis) |
| [Controllable Garment Image Synthesis Integrated with Frequency Domain Features](https://diglib.eg.org/xmlui/bitstream/handle/10.1111/cgf14938/v42i7_10_14938.pdf?sequence=1&isAllowed=y) | PG 2023 |  |
| [Toward Intelligent Interactive Design: A Generation Framework Based on Cross-domain Fashion Elements](https://dl.acm.org/doi/abs/10.1145/3581783.3612376) | ACM MM 2023 |  |
| [FashionDiff: A Controllable Diffusion Model Using Pairwise Fashion Elements for Intelligent Design](https://dl.acm.org/doi/abs/10.1145/3581783.3612127) | ACM MM 2023 |  |


## 7. Sketch Completion

- Sketch completion

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SketchGAN: Joint Sketch Completion and Recognition with Generative Adversarial Network](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_SketchGAN_Joint_Sketch_Completion_and_Recognition_With_Generative_Adversarial_Network_CVPR_2019_paper.pdf) | CVPR 2019 |  |
| [SketchHealer: A Graph-to-Sequence Network for Recreating Partial Human Sketches](https://core.ac.uk/download/pdf/334949144.pdf) | BMVC 2020 | [[code]](https://github.com/sgybupt/SketchHealer) | 
| [Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Sketch-BERT_Learning_Sketch_Bidirectional_Encoder_Representation_From_Transformers_by_Self-Supervised_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/avalonstrel/SketchBERT) |
| [Generative Sketch Healing](https://link.springer.com/article/10.1007/s11263-022-01623-7) | IJCV 2022 |  | 


- Sketch gap / connectivity detection

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Joint Gap Detection and Inpainting of Line Drawings](http://iizuka.cs.tsukuba.ac.jp/projects/inpainting/data/inpainting_cvpr2017.pdf) | CVPR 2017 | [[project]](http://iizuka.cs.tsukuba.ac.jp/projects/inpainting/en/) [[code]](https://github.com/kaidlc/CVPR2017_linedrawings) |
| [Detecting Viewer-Perceived Intended Vector Sketch Connectivity](https://www.cs.ubc.ca/labs/imager/tr/2022/SketchConnectivity/SketchConnectivity.pdf) | SIGGRAPH 2022 | [[project]](https://www.cs.ubc.ca/labs/imager/tr/2022/SketchConnectivity/)


## 8. Sketch Restoration, Retargeting and Inpainting

- Manga

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Exploiting Aliasing for Manga Restoration](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Exploiting_Aliasing_for_Manga_Restoration_CVPR_2021_paper.pdf) | CVPR 2021 | [[webpage]](http://www.cse.cuhk.edu.hk/~ttwong/papers/mangarestore/mangarestore.html) [[code]](https://github.com/msxie92/MangaRestoration) |
| [Seamless Manga Inpainting with Semantics Awareness](https://www.cse.cuhk.edu.hk/~ttwong/papers/mangainpaint/mangainpaint.html) | SIGGRAPH 2021 | [[webpage]](https://www.cse.cuhk.edu.hk/~ttwong/papers/mangainpaint/mangainpaint.html) [[code]](https://github.com/msxie92/MangaInpainting) |
| [Screentone-Preserved Manga Retargeting](https://arxiv.org/abs/2203.03396) | arxiv 22.03 |  |
| [Manga Rescreening with Interpretable Screentone Representation](https://arxiv.org/abs/2306.04114) | arxiv 23.06 |  |

- Hand-Drawn Drawings

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Restoration of Hand-Drawn Architectural Drawings using Latent Space Mapping with Degradation Generator](https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Restoration_of_Hand-Drawn_Architectural_Drawings_Using_Latent_Space_Mapping_With_CVPR_2023_paper.pdf) | CVPR 2023 |  |

## 9. Sketch and Shading

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Learning to Shadow Hand-drawn Sketches](https://arxiv.org/abs/2002.11812) | CVPR 2020 | [[project]](https://cal.cs.umbc.edu/Papers/Zheng-2020-Shade/index.html) [[code]](https://github.com/qyzdao/ShadeSketch) |
| [SmartShadow: Artistic Shadow Drawing Tool for Line Drawings](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_SmartShadow_Artistic_Shadow_Drawing_Tool_for_Line_Drawings_ICCV_2021_paper.pdf) | ICCV 2021 | [[project]](https://lllyasviel.github.io/Style2PaintsResearch/iccv2021/index.html)  |

## 10. Sketch Enhancement

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SketchMan: Learning to Create Professional Sketches](https://dl.acm.org/doi/abs/10.1145/3394171.3413720) | ACM MM 2020 | [[code]](https://github.com/LCXCUC/SketchMan2020) |
| [APISR: Anime Production Inspired Real-World Anime Super-Resolution](https://arxiv.org/abs/2403.01598) | CVPR 2024 | [[code]](https://github.com/Kiteretsu77/APISR) |

## 11. Sketch-Guided Object Localization 

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch-Guided Object Localization in Natural Images](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510528.pdf) | ECCV 2020 | [[project]](http://visual-computing.in/sketch-guided-object-localization/) [[code]](https://github.com/IISCAditayTripathi/SketchGuidedLocalization) |
| [Localizing Infinity-shaped fishes: Sketch-guided object localization in the wild](https://arxiv.org/abs/2109.11874) | arxiv 21.09 | [[code]](https://github.com/priba/sgol_wild) |
| [What Can Human Sketches Do for Object Detection?](https://arxiv.org/abs/2303.15149) | CVPR 2023 | [[project]](http://www.pinakinathc.me/sketch-detect/) |
| [Sketch-based Video Object Segmentation: Benchmark and Analysis](https://arxiv.org/abs/2311.07261) | BMVC 2023 | [[code]](https://github.com/YRlin-12/Sketch-VOS-datasets) |
| [Query-guided Attention in Vision Transformers for Localizing Objects Using a Single Sketch](https://arxiv.org/abs/2303.08784) | WACV 2024 | [[project]](https://vcl-iisc.github.io/locformer/) [[code]](https://github.com/vcl-iisc/locformer-SGOL) |

## 12. Sketch-Guided Video Synthesis

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Deep Sketch-guided Cartoon Video Inbetweening](https://ieeexplore.ieee.org/abstract/document/9314221) | TVCG 2021 | [[code]](https://github.com/xiaoyu258/Inbetweening) |

## 13. Sketch Recognition

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch-a-Net that Beats Humans](https://arxiv.org/abs/1501.07873) | BMVC 2015 | [[code]](https://github.com/yuqian1023/sketch-specific-data-augmentation) |
| [Sketch-a-Net: A Deep Neural Network that Beats Humans](https://link.springer.com/article/10.1007/s11263-016-0932-3) | IJCV 2017 | [[code]](https://github.com/yuqian1023/sketch-specific-data-augmentation) |
| [Deep Self-Supervised Representation Learning for Free-Hand Sketch](https://arxiv.org/abs/2002.00867) | TCSVT 2020 | [[code]](https://github.com/zzz1515151/self-supervised_learning_sketch) |
| [S3Net:Graph Representational Network For Sketch Recognition](https://ieeexplore.ieee.org/abstract/document/9102957/) | ICME 2020 | [[code]](https://github.com/yanglan0225/s3net) |
| [Sketchformer: Transformer-based Representation for Sketched Structure](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ribeiro_Sketchformer_Transformer-Based_Representation_for_Sketched_Structure_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/leosampaio/sketchformer) |
| [Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt](https://arxiv.org/abs/2005.09159) | CVPR 2020 | [[code]](https://github.com/avalonstrel/SketchBERT) |
| [Multi-Graph Transformer for Free-Hand Sketch Recognition](https://ieeexplore.ieee.org/abstract/document/9397867/) | TNNLS 2021 | [[code]](https://github.com/PengBoXiangShang/multigraph_transformer) |
| [Sketch-R2CNN: An RNN-Rasterization-CNN Architecture for Vector Sketch Recognition](https://scholars.cityu.edu.hk/files/73400281/Sketch_R2CNN_TVCG.pdf) | TVCG 2021 | [[code]](https://github.com/craigleili/Sketch-R2CNN) |
| [Vectorization and Rasterization: Self-Supervised Learning for Sketch and Handwriting](https://arxiv.org/abs/2103.13716) | CVPR 2021 | [[code]](https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch) |
| [SketchAA: Abstract Representation for Abstract Sketches](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_SketchAA_Abstract_Representation_for_Abstract_Sketches_ICCV_2021_paper.pdf) | ICCV 2021 |  |
| [SketchLattice: Latticed Representation for Sketch Manipulation](https://arxiv.org/abs/2108.11636) | ICCV 2021 | [[code]](https://github.com/qugank/sketch-lattice.github.io) |
| [Multi-Stage Spatio-Temporal Networks for Robust Sketch Recognition](https://ieeexplore.ieee.org/abstract/document/9740528/) | TIP 2022 |  |




## 14. Sketch Segmentation and Perceptual Grouping

- Semantic / Instance Segmentation

<table>
  <tr>
    <td><strong>Type</strong></td>
    <td><strong>Paper</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Code/Project Link</strong></td>
  </tr>
  <tr>
    <td rowspan=4"><strong>Pixelwise</strong></td>
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
    <td> <a href="https://ieeexplore.ieee.org/abstract/document/9686584">Exploring Local Detail Perception for Scene Sketch Semantic Segmentation</a> (scene-level) </td> 
    <td> TIP 2022 </td> 
    <td>  </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/abs/2312.12463">Open Vocabulary Semantic Scene Sketch Understanding</a> (scene-level) </td> 
    <td> CVPR 2024 </td> 
    <td> <a href="https://github.com/AhmedBourouis/Scene-Sketch-Segmentation">[code]</a> <a href="https://ahmedbourouis.github.io/Scene_Sketch_Segmentation/">[project]</a> </td>
  </tr>
  
  <tr>
    <td rowspan="10"><strong>Stroke-level</strong></td>
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
    <td> <a href="http://sweb.cityu.edu.hk/hongbofu/doc/fast_sketch_segmentation_CGA2019.pdf">Fast Sketch Segmentation and Labeling With Deep Learning</a> </td> 
    <td> CGA 2019 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/pdf/1901.03427.pdf">Stroke-based sketched symbol reconstruction and segmentation</a> </td> 
    <td> CGA 2020 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="http://sweb.cityu.edu.hk/hongbofu/doc/SketchGNN_TOG21.pdf">SketchGNN: Semantic Sketch Segmentation with Graph Neural Networks</a> </td> 
    <td> TOG 2021 </td> 
    <td> <a href="https://github.com/sYeaLumin/SketchGNN">[code]</a> </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/abs/2112.10838">One Sketch for All: One-Shot Personalized Sketch Segmentation</a> </td> 
    <td> TIP 2022 </td> 
    <td>  </td>
  </tr>
  <tr>
    <td> <a href="https://ieeexplore.ieee.org/abstract/document/10214525">Sketch-Segformer: Transformer-Based Segmentation for Figurative and Creative Sketches</a> </td> 
    <td> TIP 2023 </td> 
    <td> <a href="https://github.com/PRIS-CV/Sketch-SF">[code]</a> </td>
  </tr>
  <tr>
    <td> <a href="https://pubmed.ncbi.nlm.nih.gov/38470581/">CreativeSeg: Semantic Segmentation of Creative Sketches</a> </td> 
    <td> TIP 2024 </td> 
    <td> <a href="https://github.com/PRIS-CV/Sketch-CS">[code]</a> </td>
  </tr>
</table>

- Panoptic Segmentation
                                                   
<table>
  <tr>
    <td><strong>Type</strong></td>
    <td><strong>Paper</strong></td>
    <td><strong>Source</strong></td>
    <td><strong>Code/Project Link</strong></td>
  </tr>
  
  <tr>
    <td rowspan="4"><strong>Vector</strong></td>
    <td> <a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_FloorPlanCAD_A_Large-Scale_CAD_Drawing_Dataset_for_Panoptic_Symbol_Spotting_ICCV_2021_paper.pdf">FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting</a> </td> 
    <td> ICCV 2021 </td> 
    <td> <a href="https://floorplancad.github.io/">[project]</a> </td>
  </tr>
  <tr>
    <td> <a href="https://arxiv.org/abs/2201.00625">GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings</a> </td> 
    <td> CVPR 2022 </td> 
    <td> </td>
  </tr>
  <tr>
    <td> <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.pdf">CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings</a> </td> 
    <td> CVPR 2022 </td> 
    <td> <a href="https://github.com/VITA-Group/CADTransformer">[code]</a> </td>
  </tr>
  <tr>
    <td> <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_VectorFloorSeg_Two-Stream_Graph_Attention_Network_for_Vectorized_Roughcast_Floorplan_Segmentation_CVPR_2023_paper.pdf">VectorFloorSeg: Two-Stream Graph Attention Network for Vectorized Roughcast Floorplan Segmentation</a> </td> 
    <td> CVPR 2023 </td> 
    <td> <a href="https://github.com/DrZiji/VecFloorSeg">[code]</a> </td>
  </tr>
  
</table>
                                                   
- Perceptual Grouping

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


## 15. Sketch Representation Learning

- Stroke order importance/saliency, sketch abstraction

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [Learning Deep Sketch Abstraction](http://openaccess.thecvf.com/content_cvpr_2018/papers/Muhammad_Learning_Deep_Sketch_CVPR_2018_paper.pdf) | CVPR 2018 |  | Vector/stroke-level | FG-SBIR |
| [Goal-Driven Sequential Data Abstraction](http://openaccess.thecvf.com/content_ICCV_2019/papers/Muhammad_Goal-Driven_Sequential_Data_Abstraction_ICCV_2019_paper.pdf) | ICCV 2019 |  | Vector/stroke-level | Sketch recognition |
| [Pixelor: a competitive sketching AI agent. So you think you can sketch?](https://dl.acm.org/doi/pdf/10.1145/3414685.3417840) | SIGGRAPH Asia 2020 | [[Project]](http://sketchx.ai/pixelor) [[Code]](https://github.com/dasayan05/neuralsort-siggraph) | Vector/stroke-level | Sketch synthesis and recognition |
| [SketchAA: Abstract Representation for Abstract Sketches](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_SketchAA_Abstract_Representation_for_Abstract_Sketches_ICCV_2021_paper.pdf) | ICCV 2021 |  | Vector/stroke-level | Sketch recognition, FB-SBIR, sketch healing |
| [CLIPasso: Semantically-Aware Object Sketching](https://arxiv.org/abs/2202.05822) | SIGGRAPH 2022 | [[project]](https://clipasso.github.io/clipasso/) [[code]](https://github.com/yael-vinker/CLIPasso) | Vector/stroke-level |  |
| [Abstracting Sketches through Simple Primitives](https://link.springer.com/chapter/10.1007/978-3-031-19818-2_23) | ECCV 2022 | [[code]](https://github.com/ExplainableML/sketch-primitives) | Vector/stroke-level | Sketch recognition, FG-SBIR |
| [Learning Geometry-aware Representations by Sketching](https://arxiv.org/abs/2304.08204) | CVPR 2023 | [[code]](https://github.com/illhyhl1111/LearningBySketching) | Vector/stroke-level | object attribute classification, domain transfer, stroke-based generation, FG-SBIR |
| [SketchXAI: A First Look at Explainability for Human Sketches](https://arxiv.org/abs/2304.11744) | CVPR 2023 | [[project]](https://sketchxai.github.io/) | Vector/stroke-level | sketch recognition |
| [Prediction with Visual Evidence: Sketch Classification Explanation via Stroke-Level Attributions](https://ieeexplore.ieee.org/abstract/document/10194541) | TIP 2023 |  | Vector/stroke-level |  |
| [What Sketch Explainability Really Means for Downstream Tasks](https://arxiv.org/abs/2403.09480) | CVPR 2024 |  | Vector/stroke-level |  |


- Conventional Representation Learning

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [A Neural Representation of Sketch Drawings (Sketch-RNN)](https://openreview.net/pdf?id=Hy6GHpkCW) | ICLR 2018 | [[code]](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn) | Vector/stroke-level | Reconstruction and interpolation |
| [SketchHealer: A Graph-to-Sequence Network for Recreating Partial Human Sketches](https://core.ac.uk/download/pdf/334949144.pdf) | BMVC 2020 | [[code]](https://github.com/sgybupt/SketchHealer) | Vector/stroke-level | Sketch recognition, retrieval, completion and analogy |
| [Sketchformer: Transformer-based Representation for Sketched Structure](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ribeiro_Sketchformer_Transformer-Based_Representation_for_Sketched_Structure_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/leosampaio/sketchformer) | Vector/stroke-level | Sketch classification, SBIR, reconstruction and interpolation |
| [CoSE: Compositional Stroke Embeddings](https://papers.nips.cc/paper/2020/file/723e8f97fde15f7a8d5ff8d558ea3f16-Paper.pdf) | NeurIPS 2020 | [[Code]](https://github.com/eth-ait/cose) | Vector/stroke-level | Auto-completing diagrams |
| [SketchLattice: Latticed Representation for Sketch Manipulation](https://arxiv.org/abs/2108.11636) | ICCV 2021 | [[code]](https://github.com/qugank/sketch-lattice.github.io) | Lattice graph | Sketch healing and image-to-sketch synthesis |
| [SketchODE: Learning neural sketch representation in continuous time](https://openreview.net/pdf?id=c-4HSDAWua5) | ICLR 2022 | [[Project]](https://ayandas.me/sketchode) | Vector/Stroke-level | Continuous-time representation, reconstruction & interpolation |
| [Linking Sketch Patches by Learning Synonymous Proximity for Graphic Sketch Representation](https://arxiv.org/abs/2211.16841) | AAAI 2023 | [[code]](https://github.com/CMACH508/SP-gra2seq) | Vector/Stroke-level | Sketch synthesis and sketch healing |
| [SketchKnitter: Vectorized Sketch Generation with Diffusion Models](https://openreview.net/forum?id=4eJ43EN2g6l) | ICLR 2023 | [[code]](https://github.com/XDUWQ/SketchKnitter) | Vector/Stroke-level | vectorized sketch generation |
| [ChiroDiff: Modelling chirographic data with Diffusion Models](https://openreview.net/forum?id=1ROAstc9jv) | ICLR 2023 | [[Project]](https://ayandas.me/chirodiff) | Vector/Stroke-level | vectorization, de-noising/healing, abstraction |
| [Enhance Sketch Recognition's Explainability via Semantic Component-Level Parsing](https://arxiv.org/abs/2312.07875) | AAAI 2024 | [[code]](https://github.com/GuangmingZhu/SketchESC) | Vector/Stroke-level | recognition and segmentation |
| [Modelling complex vector drawings with stroke-clouds](https://openreview.net/forum?id=O2jyuo89CK) | ICLR 2024 | | Vector/Stroke-level | reconstruction and generation |

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [A Learned Representation for Scalable Vector Graphics](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lopes_A_Learned_Representation_for_Scalable_Vector_Graphics_ICCV_2019_paper.pdf) | ICCV 2019 | [[code]](https://github.com/magenta/magenta/tree/master/magenta/models/svg_vae) | SVG | Font design |
| [DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation](https://arxiv.org/abs/2007.11301) | NeurIPS 2020 | [[Code]](https://github.com/alexandre01/deepsvg)  [[Project]](https://blog.alexandrecarlier.com/deepsvg/) | SVG | Vector Graphics Animation, reconstruction and interpolation |
| [SVGformer: Representation Learning for Continuous Vector Graphics using Transformers](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_SVGformer_Representation_Learning_for_Continuous_Vector_Graphics_Using_Transformers_CVPR_2023_paper.pdf) | CVPR 2023 |  | SVG | Reconstruction, classification, interpolation, retrieval |
| [StrokeNUWA: Tokenizing Strokes for Vector Graphic Synthesis](https://arxiv.org/abs/2401.17093) | arxiv 24.01 |  | SVG | generation |


- Self-supervised or few/zero-shot learning

| Paper | Source | Code/Project Link | Data Manner | Related Applications |
| --- | --- | --- | --- | --- |
| [Sketch Fewer to Recognize More by Learning a Co-Regularized Sparse Representation](https://ieeexplore.ieee.org/abstract/document/8949551) | TCSVT 2019 |  |  | few-shot classification |
| [On Learning Semantic Representations for Large-Scale Abstract Sketches](https://ieeexplore.ieee.org/abstract/document/9274399) | TCSVT 2020 | [[Code]](https://github.com/PengBoXiangShang/EdgeMap345C_Dataset) | Bitmap and Vector | Retrieval and recognition |
| [Deep Self-Supervised Representation Learning for Free-Hand Sketch](https://arxiv.org/abs/2002.00867) | TCSVT 2020 | [[Code]](https://github.com/zzz1515151/self-supervised_learning_sketch) | Vector/stroke-level | Retrieval and recognition |
| [Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt](https://arxiv.org/abs/2005.09159) | CVPR 2020 | [[Code]](https://github.com/avalonstrel/SketchBERT) | Vector/stroke-level | Sketch recognition, retrieval, and gestalt |
| [Vectorization and Rasterization: Self-Supervised Learning for Sketch and Handwriting](https://arxiv.org/abs/2103.13716) | CVPR 2021 | [[Code]](https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch) | Both Vector and Raster-level | Recognition |


- Scene-level

| Paper | Source | Code/Project Link |
| --- | --- | --- |
| [SceneTrilogy: On Human Scene-Sketch and its Complementarity with Photo and Text](https://arxiv.org/abs/2204.11964) | CVPR 2023 |  |
          

- Implicit Neural Representations

| Paper | Source | Code/Project Link |
| --- | --- | --- |
| [SketchINR: A First Look into Sketches as Implicit Neural Representations](https://arxiv.org/abs/2403.09344) | CVPR 2024 |  |                                                                             
                                                                                       
## 16. Sketch and Visual Correspondence

- Datasets

| Name | Paper | Source | Code/Project Link | 
| --- | --- | --- | --- | 
| [CreativeFlow+](https://www.cs.toronto.edu/creativeflow/) | [Creative Flow+ Dataset](https://www.cs.toronto.edu/creativeflow/files/2596.pdf) | CVPR 2019 | [[code]](https://github.com/creativefloworg/creativeflow) |
| [ATD-12K](https://github.com/lisiyao21/AnimeInterp) | [Deep animation video interpolation in the wild](https://arxiv.org/abs/2104.02495) | CVPR 2021 | [[code]](https://github.com/lisiyao21/AnimeInterp) |
| [AnimeRun](https://lisiyao21.github.io/projects/AnimeRun) | [AnimeRun: 2D Animation Correspondence from Open Source 3D Movies](https://lisiyao21.github.io/projects/AnimeRun) | NeurIPS 2022 | [[code]](https://github.com/lisiyao21/AnimeRun) |

- Methods

| Paper | Source | Code/Project Link | 
| --- | --- | --- | 
| [Globally optimal toon tracking](https://dl.acm.org/doi/abs/10.1145/2897824.2925872) | SIGGRAPH 2016 | [[project]](https://www.cse.cuhk.edu.hk/~ttwong/papers/toontrack/toontrack.html) |
| [SketchDesc: Learning Local Sketch Descriptors for Multi-view Correspondence](http://sweb.cityu.edu.hk/hongbofu/doc/SketchDesc_TCSVT2020.pdf) | TCSVT 2020 |  | 
| [SketchZooms: Deep Multi-view Descriptors for Matching Line Drawings](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14197) | CGF 2021 | [[project]](https://emmanueliarussi.github.io/index_sz.html) | 
| [The Animation Transformer: Visual Correspondence via Segment Matching](https://openaccess.thecvf.com/content/ICCV2021/papers/Casey_The_Animation_Transformer_Visual_Correspondence_via_Segment_Matching_ICCV_2021_paper.pdf) | ICCV 2021 |  | 
| [Differential Operators on Sketches via Alpha Contours](http://www-labs.iro.umontreal.ca/~bmpix/pdf/SketchLaplacian.pdf) | SIGGRAPH 2023 | [[code]](https://github.com/bmpix/AlphaContours) | 
| [Learning Inclusion Matching for Animation Paint Bucket Colorization](https://arxiv.org/abs/2403.18342) | CVPR 2024 | [[code]](https://github.com/ykdai/BasicPBC) [[project]](https://ykdai.github.io/projects/InclusionMatching) | 
                                                                                       

## 17. Sketch Animation/Inbetweening

- Inbetweening

| Paper | Source | Representation | Code/Project Link |
| --- | --- | :--: | --- |
| [BetweenIT: An Interactive Tool for Tight Inbetweening](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2009.01630.x) | CGF 2010 | stroke |  |
| [Context-Aware Computer Aided Inbetweening](https://ieeexplore.ieee.org/abstract/document/7831370) | TVCG 2017 | stroke |  |
| [FTP-SC: Fuzzy Topology Preserving Stroke Correspondence](https://dcgi.fel.cvut.cz/home/sykorad/Yang18-SCA.pdf) | SCA 2018 | stroke | [[webpage]](https://dcgi.fel.cvut.cz/home/sykorad/FTP-SC.html) [[video]](https://youtu.be/3oZfCAkYJQk) |
| [Cacani: 2d animation and inbetween software](https://cacani.sg/?v=1c2903397d88) | / | stroke | [[software]](https://cacani.sg/?v=1c2903397d88) |
| [Stroke-Based Drawing and Inbetweening with Boundary Strokes](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14433) | CGF 2021 | stroke |  |
| [Joint Stroke Tracing and Correspondence for 2D Animation](https://dl.acm.org/doi/10.1145/3649890) | TOG 2024 | stroke | [[code]](https://github.com/MarkMoHR/JoSTC) [[webpage]](https://markmohr.github.io/JoSTC/) |
|    |
| [Deep Geometrized Cartoon Line Inbetweening](https://openaccess.thecvf.com/content/ICCV2023/papers/Siyao_Deep_Geometrized_Cartoon_Line_Inbetweening_ICCV_2023_paper.pdf) | ICCV 2023 | vertex/point | [[code]](https://github.com/lisiyao21/AnimeInbet) |
|    |
| [Optical Flow Based Line Drawing Frame Interpolation Using Distance Transform to Support Inbetweenings](https://ieeexplore.ieee.org/abstract/document/8803506) | ICIP 2019 | raster |  |
| [Bridging the Gap: Fine-to-Coarse Sketch Interpolation Network for High-Quality Animation Sketch Inbetweening](https://arxiv.org/abs/2308.13273) | arxiv 23.08 | raster |  |

- Animation

| Paper | Source | Representation | Code/Project Link  |
| --- | --- | :--: | --- |
| [Autocomplete Hand-drawn Animations](http://junxnui.github.io/research/siga15_autocomplete_handdrawn_animations.pdf) | SIGGRAPH Asia 2015 | stroke | [[webpage]](https://iis-lab.org/research/autocomplete-animations/) [[video]](https://youtu.be/w0YmWiy6sA4) |
| [Live Sketch: Video-driven Dynamic Deformation of Static Drawings](http://sweb.cityu.edu.hk/hongbofu/doc/livesketch_CHI2018.pdf) | CHI 2018 | vector | [[video]](https://youtu.be/6DjQR5k286E) |
| [Animating Portrait Line Drawings from a Single Face Photo and a Speech Signal](https://dl.acm.org/doi/abs/10.1145/3528233.3530720) | SIGGRAPH 2022 | image | [[code]](https://github.com/AnimatePortrait/AnimatePortrait) |
| [A Method for Animating Children’s Drawings of the Human Figure](https://arxiv.org/abs/2303.12741) | TOG 2023 | image | [[code]](https://github.com/facebookresearch/AnimatedDrawings) [[project]](https://people.csail.mit.edu/liyifei/publication/children-animated-drawings/) [[demo]](https://sketch.metademolab.com/canvas) |
| [Non-linear Rough 2D Animation using Transient Embeddings](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14771) | CGF 2023 | stroke |  |
| [Sketch Video Synthesis](https://arxiv.org/abs/2311.15306) | arxiv 23.11 | stroke | [[code]](https://github.com/yudianzheng/SketchVideo) [[project]](https://sketchvideo.github.io/) |
| [Breathing Life Into Sketches Using Text-to-Video Priors](https://arxiv.org/abs/2311.13608) | CVPR 2024 | stroke | [[code]](https://github.com/yael-vinker/live_sketch) [[project]](https://livesketch.github.io/) |


## 18. Sketch and AR/VR

| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [SweepCanvas: Sketch-based 3D Prototyping on an RGB-D Image](http://sweb.cityu.edu.hk/hongbofu/doc/sweep_canvas_UIST2017.pdf) | UIST 2017 | [[video]](https://youtu.be/Xnp3_eMYXj0) |
| [Model-Guided 3D Sketching](http://sweb.cityu.edu.hk/hongbofu/doc/model-guided_3D_sketching_TVCG.pdf) | TVCG 2018 | [[video]](https://youtu.be/STredKjB_Bk) |
| [Mobi3DSketch: 3D Sketching in Mobile AR](http://sweb.cityu.edu.hk/hongbofu/doc/mobi3Dsketch_CHI2019.pdf) | CHI 2019 | [[video]](https://youtu.be/JdP0nkeMEog) |
| [Interactive Liquid Splash Modeling by User Sketches](https://dl.acm.org/doi/abs/10.1145/3414685.3417832) | SIGGRAPH Asia 2020 | [[video]](https://youtu.be/HXAxNrfk_w0) |
| [Towards 3D VR-Sketch to 3D Shape Retrieval](https://rowl1ng.com/assets/pdf/3DV_VRSketch.pdf) | 3DV 2020 | [[code]](https://github.com/ygryadit/Towards3DVRSketch) [[project]](https://rowl1ng.com/projects/3DSketch3DV/) |
| [3D Curve Creation on and around Physical Objects with Mobile AR](http://sweb.cityu.edu.hk/hongbofu/doc/3D_Curve_Creation_Mobile_AR_TVCG.pdf) | TVCG 2021 | [[video]](https://youtu.be/zyh4pEvK7j8) |
| [HandPainter - 3D Sketching in VR with Hand-based Physical Proxy](https://dl.acm.org/doi/abs/10.1145/3411764.3445302) | CHI 2021 | [[video]](https://youtu.be/x5VAU-471P8) |
| [Fine-Grained VR Sketching: Dataset and Insights](https://ieeexplore.ieee.org/abstract/document/9665875/) | 3DV 2021 | [[code]](https://github.com/Rowl1ng/Fine-Grained_VR_Sketching) |
| [Structure-Aware 3D VR Sketch to 3D Shape Retrieval](https://github.com/Rowl1ng/Structure-Aware-VR-Sketch-Shape-Retrieval) | 3DV 2022 | [[code]](https://github.com/Rowl1ng/Structure-Aware-VR-Sketch-Shape-Retrieval) |
| [GestureSurface: VR Sketching through Assembling Scaffold Surface with Non-Dominant Hand](https://ieeexplore.ieee.org/abstract/document/10049645) | TVCG 2023 |  |
| [3D VR Sketch Guided 3D Shape Prototyping and Exploration](https://arxiv.org/abs/2306.10830) | ICCV 2023 | [[code]](https://github.com/Rowl1ng/3Dsketch2shape) |


## 19. Sketch Based Incremental Learning
| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Doodle It Yourself: Class Incremental Learning by Drawing a Few Sketches](https://arxiv.org/abs/2203.14843) | CVPR 2022 | [[code]](https://github.com/AyanKumarBhunia/DIY-FSCIL) |

                                                                                       
## 20. Sketch Quality Measurement
| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Annotation-Free Human Sketch Quality Assessment](https://link.springer.com/article/10.1007/s11263-024-02001-1) | IJCV 2024 | [[code]](https://github.com/yanglan0225/SketchX-Quantifying-Sketch-Quality) |
| [Finding Badly Drawn Bunnies](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Finding_Badly_Drawn_Bunnies_CVPR_2022_paper.pdf) | CVPR 2022 | [[code]](https://github.com/yanglan0225/SketchX-Quantifying-Sketch-Quality) |

## 21. Cloud Augmentation with Sketches
| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Cloud2Sketch: Augmenting Clouds with Imaginary Sketches](https://dl.acm.org/doi/abs/10.1145/3503161.3547810) | ACM MM 2022 | [[project]](https://wanzy.me/research/cloud2sketch) |

## 22. Sketch and Re-identification
| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Cross-Compatible Embedding and Semantic Consistent Feature Construction for Sketch Re-identification](https://dl.acm.org/doi/abs/10.1145/3503161.3548224) | ACM MM 2022 | [[code]](https://github.com/lhf12278/CCSC) |
| [SketchTrans: Disentangled Prototype Learning with Transformer for Sketch-Photo Recognition](https://ieeexplore.ieee.org/document/10328884) | TPAMI 2023 | [[code]](https://github.com/ccq195/SketchTrans) |

## 23. Sketch-based Salient Object Detection
| Paper | Source | Code/Project Link  |
| --- | --- | --- |
| [Sketch2Saliency: Learning to Detect Salient Objects from Human Drawings](https://arxiv.org/abs/2303.11502) | CVPR 2023 |  |
