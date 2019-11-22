# 基于GANs的动漫面孔生成 #

DCGAN:效果较差，96x96  
StyleGAN:效果较好，256x256  
StyleGAN_迁移特定角色:尝试迁移了Bang Dream第一季（以此为数据集），效果不错~
交互控制生成和修改:用参考最后一条产生的标签加上逻辑回归分类产生了生成蓝色头发的方向向量（其他有机会试试）

emmm本来是可以有展示的可惜学校不给公网IP，如果真的有人看到了需要模型、向量等训练出的文件再说吧emmm

## 参考 ##
<https://github.com/NVlabs/stylegan>  
<https://github.com/carpedm20/DCGAN-tensorflow>  
<https://github.com/mattya/chainer-DCGAN>  
<https://github.com/nagadomi/lbpcascade_animeface>用于截取动漫面孔  
<https://github.com/nagadomi/waifu2x>用于放大分辨率过低的数据  
<https://qiita.com/mattya/items/e5bfe5e04b9d2f0bbd47>  
<https://www.gwern.net/Faces>  
<https://zhuanlan.zhihu.com/p/63230738>  
<http://www.seeprettyface.com/index.html>  
<https://blog.csdn.net/weixin_43013761/article/details/100895333>  
<https://zhuanlan.zhihu.com/p/57553117>  
<https://www.reddit.com/r/MachineLearning/comments/akbc11/p_tag_estimation_for_animestyle_girl_image/>确定生成图的标签