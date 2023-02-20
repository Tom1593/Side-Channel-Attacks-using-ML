to run the code you need to change a few things:
1.you need to change the working directory and we did this by hand, param = directory
2.need to change param = save_str ->results directory
3.we added an option to enable feature selection, param = feat_select
4.we added an option to enable or disable the padding feature, param = feat_select
5.we added a low pass filter to deal with the noise but we had bad results with low noise so why use it when you dont need it
