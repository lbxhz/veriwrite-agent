# 气溶胶遥感反演方法、应用与局限

气溶胶是影响全球气候变化、环境质量和人体健康的重要因素（Liu et al., 2021; Cohen et al., 2017）。准确获取气溶胶光学厚度（Aerosol Optical Depth, AOD）的时空分布，是定量评估其辐射效应与健康影响的关键前提。近年来，卫星遥感已成为大尺度气溶胶监测的主要手段，其反演方法经历了从物理模型到数据驱动方法的显著演进。

传统卫星AOD反演主要依赖辐射传输模型（RTM）和查找表（LUT）技术。MODIS系列产品（如MOD04_L2和MCD19A2）以及Himawari-8/AHI等静止轨道卫星产品均基于此原理（Levy et al., 2015; Lyapustin et al., 2018）。这类方法通过预先计算的查找表建立卫星观测辐射与AOD之间的映射关系。然而，RTM算法常需简化气溶胶模型和地表反射率假设，校准误差、云掩膜不完善及查找表插值误差均会引入不确定性（Grey et al., 2006; Levy et al., 2010）。Chen et al.（2022）指出，基于CERES和MODIS数据的经验拟合方法可避免辐射传输模型的复杂假设，其依据Beer-Lambert定律将晴空辐射通量拟合为AOD的指数函数，在中纬度地区（20°N–40°N）取得了良好效果，地表和大气顶的拟合优度R²分别达0.98和0.995。

为克服传统方法的局限，机器学习方法被广泛引入AOD反演。Song et al.（2025）利用Geo-KOMPSAT-2A（GK-2A）静止卫星数据，分别构建了基于红外亮温的全天候模型和基于大气顶反射率的白天模型，采用随机森林和LightGBM算法估算东亚地区AOD。结果显示，白天TOA模型性能略高于全天候BT模型（R²=0.83, RMSE=0.098），而全天候BT模型首次实现了夜间AOD的连续估算，弥补了静止卫星夜间观测的空白。SHAP分析表明，总可降水量和季节因子是影响模型输出的最关键变量。这一发现与Sun et al.（2026）的研究相呼应——后者提出了一种两阶段混合降尺度算法，利用集成机器学习校准MERRA-2再分析AOD，并结合改进的Downscale-UNet深度学习模型（引入卷积块注意力模块和深度可分离卷积）生成中国地区无缝1km分辨率AOD产品。独立验证表明，该方法相对于MERRA-2精度提升56%，与MCD19A2卫星产品精度相当（R²=0.75, RMSE=0.11）。

在应用层面，AOD反演产品被广泛用于气溶胶直接辐射强迫（ADRF）评估。Chen et al.（2022）基于MODIS和CERES数据估算的北半球中纬度区域平均ADRF在地表、大气顶和大气层内分别为-6.6、-2.5和4.0 W/m²，表明气溶胶在地表和大气顶产生冷却效应，在大气层内产生增温效应。研究还发现，2000–2009年间AOD以每年0.0029的速率增加，而2010–2019年间则以每年-0.0039的速率下降，相应的ADRF变化与AOD趋势高度一致，反映出中国东部、印度北部等地区人为排放控制的成效。

然而，各类方法仍存在显著局限。物理反演方法的空间覆盖受云、雪和亮地表影响，存在大量数据空缺（Christopher and Gupta, 2010）。以MCD19A2为例，在中国区域仅有约44%的有效空间覆盖（Sun et al., 2026）。机器学习方法虽可改善精度和覆盖度，但其性能高度依赖于训练样本的质量与分布。Song et al.（2025）指出，AERONET站点在蒙古、中国内陆及开阔海域分布稀疏，限制了模型在这些区域的表征能力。此外，机器学习方法缺乏明确的物理机制，对高值AOD存在系统性低估。再分析资料（如MERRA-2）虽然覆盖率完整，但空间分辨率较粗（约50km），难以捕捉局地尺度气溶胶变化（Sun et al., 2026）。未来研究应致力于发展物理约束与数据驱动相融合的反演框架，并拓展夜间及全天候观测能力，以全面支撑气溶胶气候效应评估与空气质量监测。


## 参考文献

Chen, A., Zhao, C., Fan, T., 2022. Spatio-temporal distribution of aerosol direct radiative forcing over mid-latitude regions in north hemisphere estimated from satellite observations. Atmospheric Research, 266, 105938. https://doi.org/10.1016/j.atmosres.2021.105938

Song, S., Kang, Y., Im, J., Park, S.S., 2025. Enhanced continuous aerosol optical depth (AOD) estimation using geostationary satellite data: focusing on nighttime AOD over East Asia. Atmospheric Environment, 358, 121365. https://doi.org/10.1016/j.atmosenv.2025.121365

Sun, L., Zhang, X., Fan, Y., Wang, Z., Sun, X., 2026. Downscaling aerosol optical depth by fusing satellite retrieval and model simulation using artificial intelligence technology. Atmospheric Research, 328, 108411. https://doi.org/10.1016/j.atmosres.2025.108411