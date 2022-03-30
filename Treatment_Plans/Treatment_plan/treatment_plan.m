% Set configurations
clear;
% visit_dates = readtable('/exports/lkeb-hpc/tlandman/Patient_Data/visits.csv', 'Delimiter', ',');
% visit_dates = table2cell(visit_dates);
% patient_info.ID = visit_dates(:,1);
% patient_info.visit_date = visit_dates(:,2);
patient_info.contours = {'GTV', 'SeminalVesicle', 'Rectum', 'Bladder', 'Torso'}; %'ITVsubSV',

%% Load CT image

for i = 84 %156:length(patient_info.visit_date)
%     disp(strcat('/exports/lkeb-hpc/tlandman/Patient_Data/', patient_info.ID{i,1}, '/', patient_info.visit_date{i,1}, '/CTImage.nrrd'))
%     [ct.cubeHU{1,1},info] = matRad_readNRRD(strcat('/exports/lkeb-hpc/tlandman/Patient_Data/', patient_info.ID{i,1}, '/', patient_info.visit_date{i,1}, '/CTImage.nrrd'));
    [ct.cubeHU{1,1},info] = matRad_readNRRD(strcat('/exports/lkeb-hpc/tlandman/Data/Patient_NRRD/Patient_22/visit_20071108/CTImage.nrrd'));
    ct.cubeHU{1,1} = double(ct.cubeHU{1,1});
    ct.cubeDim = info.cubeDim;
    ct.resolution.x = info.resolution(1);
    ct.resolution.y = info.resolution(2);
    ct.resolution.z = info.resolution(3);
    ct.numOfCtScen = 1;
    clear info;

%% Load segmentations
    % Now we define structures a contour for the phantom and a target
    colors = {[0.85,0.85,0.85], [0.7,0.7,0.7], [0,0,0], [0.75,0,0.75], [1,1,0]}; %[1,1,1]
    %cst = cell(length(patient_info.contours),7);
    for ix = 1:length(patient_info.contours)

        cst{ix,1} = ix-1;

        cst{ix,2} = patient_info.contours{ix};

        if ix <= 2
            cst{ix,3} = 'TARGET';
        else
            cst{ix,3} = 'OAR';
        end

%         [voxel_cube, ~] = matRad_readNRRD(strcat('/exports/lkeb-hpc/tlandman/Patient_Data/', patient_info.ID{i,1}, '/', patient_info.visit_date{i,1}, '/masks/',patient_info.contours{ix},'.nrrd'));
        [voxel_cube, ~] = matRad_readNRRD(strcat('/exports/lkeb-hpc/tlandman/Data/Patient_NRRD/Patient_22/visit_20071108/masks/',patient_info.contours{ix},'.nrrd'));cst{ix, 4}{1} = find(voxel_cube);

        cst{ix,5}.TissueClass  = 1;
        cst{ix,5}.alphaX       = 0.1000;
        cst{ix,5}.betaX        = 0.0500;
        cst{ix,5}.Priority     = ix;
        cst{ix,5}.Visible      = 1;
        cst{ix,5}.visibleColor = colors{1,ix};

    end
    clear ix colors voxel_cube;

    % adjust the constraints / objective

%     cst{1,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDose(0.94*74,1.5*74,'approx'));
%     cst{2,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDose(0.94*55,1.5*55,'approx'));
    %cst{2,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDose(0.99*55,1.08*55,'approx'));

    cst{1,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDVH(0.97*74, 98, 100));
    cst{1,6}{1,2} = struct(DoseObjectives.matRad_MaxDVH(1000,1.05*74,2));

    cst{2,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDVH(0.97*55, 98, 100));
    cst{2,6}{1,2} = struct(DoseObjectives.matRad_MaxDVH(1000,1.05*55,2));

    %cst{1,6}{1,3} = struct(DoseObjectives.matRad_MeanDose(500,74));

    %cst{2,6}{1,2} = struct(DoseObjectives.matRad_SquaredOverdosing(1000,1.06*55));
    %cst{1,6}{1,3} = struct(DoseObjectives.matRad_SquaredUnderdosing(1000,0.97*55));
    %cst{2,6}{1,3} = struct(DoseObjectives.matRad_SquaredUnderdosing(1000,0.99*55));

    %     cst{3,6}{1,1} = struct(DoseObjectives.matRad_MaxDVH(200,1.02*74,0.1));
    %     cst{4,6}{1,1} = struct(DoseObjectives.matRad_MaxDVH(200,1.02*74,0.1));
    %     cst{5,6}{1,1} = struct(DoseObjectives.matRad_MaxDVH(200,1.06*74,0.1));
    %     
    %     cst{3,6}{1,1} = struct(DoseObjectives.matRad_MeanDose(10,0));
    %     cst{4,6}{1,1} = struct(DoseObjectives.matRad_MeanDose(10,0));
    %cst{5,6}{1,1} = struct(DoseObjectives.matRad_SquaredOverdosing(1000,1.06*74));

    %cst{3,6}{1,2} = struct(DoseObjectives.matRad_MeanDose(10,0));
    %cst{4,6}{1,2} = struct(DoseObjectives.matRad_MeanDose(10,0));


%% Treatment Plan
    % The next step is to define the treatment plan labeled as 'pln'. This 
    % structure requires input from the treatment planner and defines 
    % the most important cornerstones of your treatment plan.

    pln.radiationMode           = 'protons';           
    pln.machine                 = 'Generic';
    pln.numOfFractions          = 1;
    pln.propOpt.bioOptimization = 'const_RBExD';     
    pln.propStf.gantryAngles    = [90 270];
    pln.propStf.couchAngles     = [0 0];
    pln.propStf.bixelWidth      = 2;
    pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
    pln.propStf.isoCenter       = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);
    pln.propOpt.runDAO          = 0;
    pln.propOpt.runSequencing   = 0;

    % dose calculation settings
    pln.propDoseCalc.doseGrid.resolution.x = 2; % [mm]
    pln.propDoseCalc.doseGrid.resolution.y = 2; % [mm]
    pln.propDoseCalc.doseGrid.resolution.z = 2; % [mm]

%% Generate Beam Geometry STF
    stf = matRad_generateStf(ct,cst,pln);

%% Dose Calculation
    dij = matRad_calcParticleDose(ct,stf,pln,cst);

%% Optimize the plan
    resultGUI = matRad_fluenceOptimization(dij,cst,pln);
        
%% DVH 
    [dvh,qi]              = matRad_indicatorWrapper(cst,pln,resultGUI,[45,52,59,70,80],[2,50]);
    
    
%% Save the dose as a DICOM file
    dcmObj = matRad_DicomExporter;              % create instance of matRad_DicomExporter
    visit_dir_path = '/exports/lkeb-hpc/tlandman/Data/Patient_DCM/Patient_22/visit_20071108/'; %/exports/lkeb-hpc/tlandman/temp/plan'; % strcat('/exports/lkeb-hpc/tlandman/Patient_Dose/', patient_info.ID{i,1}, '/', patient_info.visit_date{i,1});
%     pat_dir_path = strcat('/exports/lkeb-hpc/tlandman/Patient_Dose/', patient_info.ID{i,1});
%     if ~exist(pat_dir_path, 'dir')
%         mkdir(pat_dir_path)
%     end
    
    visit_dir_path = convertCharsToStrings(visit_dir_path);
    
    if ~exist(visit_dir_path, 'dir')
%         disp('hoi');
        mkdir(visit_dir_path)
        mkdir(visit_dir_path+'/CT')
        mkdir(visit_dir_path+'/Struct')
        mkdir(visit_dir_path+'/Dose')
    end
%%
    dcmObj.dicomDir = visit_dir_path;           % set the output path for the Dicom export
    dcmObj.matRad_exportDicom();                % run the export
%% Clear the parameters
%     clear ct cst stf dij resultGUI
end