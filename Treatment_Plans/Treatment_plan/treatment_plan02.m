cst{1,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDose(0.97*74,1.2*74,'approx'));
cst{2,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDose(0.99*55,1.5*55,'approx'));
%cst{2,6}{1,1} = struct(DoseConstraints.matRad_MinMaxDose(0.99*55,1.08*55,'approx'));

%cst{1,6}{1,2} = struct(DoseObjectives.matRad_SquaredOverdosing(1000,1.06*74));

%cst{2,6}(:,2) = []; % = [];  % struct(DoseObjectives.matRad_SquaredOverdosing(1000,1.06*74));
%cst{2,6}(:,2) = [];  % struct(DoseObjectives.matRad_MeanDose(500,74));

cst{1,6}{1,2} = struct(DoseObjectives.matRad_SquaredOverdosing(1000,1.1*55));
cst{1,6}{1,3} = struct(DoseObjectives.matRad_MeanDose(500,55));
%cst{1,6}{1,3} = struct(DoseObjectives.matRad_SquaredUnderdosing(1000,0.97*55));
%cst{2,6}{1,3} = struct(DoseObjectives.matRad_SquaredUnderdosing(1000,0.99*55));

%cst{3,6}{1,1} = struct(DoseObjectives.matRad_SquaredOverdosing(40,1.02*74));
%cst{4,6}{1,1} = struct(DoseObjectives.matRad_SquaredOverdosing(40,1.02*74));
%cst{5,6}{1,1} = struct(DoseObjectives.matRad_SquaredOverdosing(40,1.06*74));

%cst{3,6}{1,1} = struct(DoseObjectives.matRad_MeanDose(10,0));
%cst{4,6}{1,1} = struct(DoseObjectives.matRad_MeanDose(10,0));
%% Optimize the plan
dij2 = dij;
resultGUI               = matRad_fluenceOptimization(dij2,cst,pln);
resultGUI.RBExDose = resultGUI.RBExDose .* 30;
resultGUI.RBExDose_beam1 = resultGUI.RBExDose_beam1 .* 30;
resultGUI.RBExDose_beam2 = resultGUI.RBExDose_beam2 .* 30;
resultGUI.physicalDose = resultGUI.physicalDose .* 30;
resultGUI.physicalDose_beam1 = resultGUI.physicalDose_beam1 .* 30;
resultGUI.physicalDose_beam2 = resultGUI.physicalDose_beam2 .* 30;

%% Show some results
[dvh,qi]              = matRad_indicatorWrapper(cst,pln,resultGUI,[45,52,59,70,80],[2,50]);