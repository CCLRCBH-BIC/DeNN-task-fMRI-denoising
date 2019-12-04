function [X,ref,ref_conv,deltaT] = HCP_blockDesignmatrix(eprime_file,sync_file)
TR=0.72;tdim = 405;
OFFSETtime = importdata(sync_file);
% OFFSETtime is the sync time;
A = importdata(eprime_file);
RAW = A.textdata;
row1 = find(strcmp(RAW(:,1),'ExperimentName')==1,1);

%find column with SlideC.onsetTime
ind_stim = find(strcmp(RAW(row1,:),'Stim.OnsetTime') == 1);
ind_blocktype = find(strcmp(RAW(row1,:),'BlockType') == 1);
ind_Fix_15 = find(strcmp(RAW(row1,:),'Fix15sec.OnsetTime') == 1);

%%%%%%%%%%%%%%%%%%%%find onset for all the stimuli%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cell0bk = (RAW(strcmp(RAW(:,ind_blocktype),'0-Back'),ind_stim));
cell2bk = (RAW(strcmp(RAW(:,ind_blocktype),'2-Back'),ind_stim));
cell_fix = (RAW(row1+1:end,ind_Fix_15));
cell_fix = unique((cell_fix));
N0bk = numel(cell0bk);
N2bk = numel(cell2bk);
Nfix = numel(cell_fix);
start0bk = zeros(N0bk,1);
for i = 1:N0bk
    start0bk(i,1) = str2double(cell0bk{i})/1000-OFFSETtime;
end

start2bk = zeros(N2bk,1);
for i = 1:N2bk
    start2bk(i,1) = str2double(cell2bk{i})/1000-OFFSETtime;
end

startfix = zeros(Nfix,1);
for i = 1:Nfix
    if ~isempty(cell_fix{i})
        startfix(i,1) = str2double(cell_fix{i})/1000-OFFSETtime;
    end
end
startfix = startfix(startfix~=0);
Nfix = numel(startfix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nr = 3;
deltaT = 0.1;
Nt = round(TR*tdim/deltaT);
ref = zeros(Nt,Nr);
for j = 1:N0bk
    ref(round(start0bk(j,1)/deltaT):round((start0bk(j,1)+2.5)/deltaT),1) = 1;
end
for j = 1:N2bk
    ref(round(start2bk(j,1)/deltaT):round((start2bk(j,1)+2.5)/deltaT),2) = 1;
end
for j = 1:Nfix
    if (round(startfix(j,1)+15)/deltaT)>Nt
        ref(round(startfix(j,1)/deltaT):Nt,3) = 1;
    else
        ref(round(startfix(j,1)/deltaT):round((startfix(j,1)+15)/deltaT),3) = 1;
    end
end
% imagesc(ref)

canonHRF = spm_hrf(deltaT); tdim_hrf = numel(canonHRF);
ref_conv = zeros(Nt+tdim_hrf-1,Nr);
for i = 1:Nr
    ref_conv(:,i) = conv(ref(:,i),canonHRF);
end
ref_conv = ref_conv(1:Nt,:);
TRarray = TR*(1:tdim)';
deltatarray = deltaT*(1:Nt)';
X = interp1(deltatarray,ref_conv,TRarray,'PCHIP','extrap');
% figure,imagesc(X);
end
