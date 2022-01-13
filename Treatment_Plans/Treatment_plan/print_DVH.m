for i = 2:10
    GTV_V95(i-1) = getfield(qi{1,i}(1),'V_70_3Gy');
    GTV_V107(i-1) = getfield(qi{1,i}(1),'V_79_2Gy');
    GTV_V110(i-1) = getfield(qi{1,i}(1),'V_81_4Gy');
    SV_V95(i-1) = getfield(qi{1,i}(2),'V_52_2Gy');
    SV_V107(i-1) = getfield(qi{1,i}(2),'V_58_9Gy');
    SV_V110(i-1) = getfield(qi{1,i}(2),'V_60_5Gy');
    
    
end
disp(mean(GTV_V95))
disp(mean(GTV_V107))
disp(mean(GTV_V110))
disp(mean(SV_V95))
disp(mean(SV_V107))
disp(mean(SV_V110))