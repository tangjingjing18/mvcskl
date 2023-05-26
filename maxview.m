function max_view = maxview(A,B,weightAB,halfAB)
%%%输出某一指标下最大的指标值，对应的标准差，以及对应的视角的ID
    [mean_a,mean_b,mean_w,mean_h] = deal(nanmean(A),nanmean(B),nanmean(weightAB),nanmean(halfAB));  
    [std_a,std_b,std_w,std_h] = deal(nanstd(A),nanstd(B),nanstd(weightAB),nanstd(halfAB)); 
    std = [std_a,std_b,std_w,std_h];
        
    [max_allview,view_id] = max([mean_a,mean_b,mean_w,mean_h]);%ID和视角的对应
    std_allview = std(view_id);
    max_view.value = max_allview;
    max_view.std = std_allview;
    max_view.id = view_id;
end
    