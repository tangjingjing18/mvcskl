clc;
clear;
num_string = [];
namelist = dir('./data/*.mat');
save_path = './results/';
for i = 1:length(namelist)
    data_path=strcat('./data/',namelist(i).name);
    string = namelist(i).name;
    data_num = string(1:end-4);
    [result] = mvlinex_corel_run(data_path,save_path,data_num);
    resultarr(i).name = data_num;
    resultarr(i).output = result;
end