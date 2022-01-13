visit_dates = csvread('visit_dates.csv');
patient_info.ID = string(visit_dates(:,1));
patient_info.visit_date = string(visit_dates(:,2));
for i = 1:10
    disp(strcat('/exports/lkeb-hpc/tlandman/Patient_Data/Patient_0', patient_info.ID(i), '/visit_', patient_info.visit_date(i), '/CTImage.nrrd'))
end
disp(patient_info.visit_date(3))