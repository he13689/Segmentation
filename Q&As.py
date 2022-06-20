"""
    ViT中的 cls token 是什么用？
        # 因为transformer输入为一系列的patch embedding，输出也是同样长的序列patch feature，
        # 但是最后要总结为一个类别的判断，简单方法可以用avg pool，把所有的patch feature都考虑算出image feature
        # 但是作者没有用这种方式，而是引入一个类似flag的class token,其输出特征加上一个线性分类器就可以实现分类
        # 其中训练的时候，class token的embedding被随机初始化并与pos embedding相加

        ViT做分类时取出第n+1个token作为分类的特征，这样做的原理在哪里?
        有人说这样是为了避免对输入的某一个token有偏向性，那么我将前n个token做平均作为要分类的特征是否可行呢？
            首先不存在n+1这个意思奥，论文里面是class token是放在首位，也就是第0个位置，
            第n+1个token（class embedding）的主要特点是：（1）不基于图像内容；（2）位置编码固定。
                1、该token随机初始化，并随着网络的训练不断更新，它能够编码整个数据集的统计特性；
                2、该token对所有其他token上的信息做汇聚（全局特征聚合），并且由于它本身不基于图像内容，因此可以避免对sequence中某个特定token的偏向性；
                3、对该token使用固定的位置编码能够避免输出受到位置编码的干扰。ViT中作者将class embedding视为sequence的头部而非尾部，即位置为0。
                    这样即使sequence的长度n发生变化，class embedding的位置编码依然是固定的，因此，更准确的来说class embedding应该是第0个而非第n+1个token。
                    另外题主说的“将前n个token做平均作为要分类的特征是否可行呢”，这也是一种全局特征聚合的方式，但它相较于采用attention机制来做全局特征聚合而言表达能力较弱。
                    因为采用attention机制来做特征聚合，能够根据query和key之间的关系来自适应地调整特征聚合的权重，
                    而采用求平均的方式则是对所有的key给了相同的权重，这限制了模型的表达能力。


"""