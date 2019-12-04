clear
clc
datadir = 'F:\HCP_test1_unzipped';
[num,txt,raw] = xlsread('F:\HCP_test1_tables\unrestricted_zhengshi_9_10_2018_12_6_45.csv');
% designStrCell = {'2bk_cor.txt','0bk_cor.txt'};
TR = 0.72;
FWHM = 0;

analysis_method = {'GLM','FIX40','CONFreg','motreg','WMCSF','TF'};
compcor_num = 5;
for i = 2:size(raw,1)
    subdir = [datadir,'\',num2str(raw{i,1})]
    if exist(subdir,'dir')==0
        continue;
    end
    analysisdir = [subdir,'\WM_fMRI\analysis'];
    delete([analysisdir,'\GLM*.nii']);
    
    mkdir(analysisdir);
    eprime_file = [subdir,'\WM_fMRI\WM_run2_TAB.txt'];
    sync_file = [subdir,'\WM_fMRI\EVs\Sync.txt'];
    
    [X_tgtlure,ref_TR,X_deltaT,ref_deltaT]=HCP_tgtlureDesignmatrix(eprime_file,sync_file);
    contrast_tgtlure = [1;0;-1];
    [X_block,ref,ref_conv,deltaT] = HCP_blockDesignmatrix(eprime_file,sync_file);
    contrast_block = [-1;1;0];
    for a=1%1:numel(analysis_method)
        analysis = analysis_method{a};
        if strcmpi(analysis,'GLM')==1
            fMRIdatanii = load_untouch_nii([subdir,'\WM_fMRI\tfMRI_WM_LR.nii.gz']);fMRIdata = fMRIdatanii.img;
            mask = squeeze(fMRIdata(:,:,:,1))>0;
            fMRIdata = permute(fMRIdata,[4,1,2,3]);
            if FWHM>0
                fMRIdata = imfilter(fMRIdata,f_origin,'same');
                fMRIdata(:,mask<=0) = 0;
            end
            fMRIdata = fMRIdata(16:end,mask>0);
            
            %%%%extract data for LSTMdenoise
            c1T1 = load_untouch_nii([subdir,'\T1\rc1T1w_restore_brain.nii']);c1 = c1T1.img;
            c2T1 = load_untouch_nii([subdir,'\T1\rc2T1w_restore_brain.nii']);
            c3T1 = load_untouch_nii([subdir,'\T1\rc3T1w_restore_brain.nii']);
            c23 = c2T1.img+c3T1.img;
            
            [fsteer,gsteer,f_origin]=steerfilter(4,3,[2 2 2 3 3 3]);
            f_origin = reshape(f_origin,[1 size(f_origin)]);
            
            sc23 = imfilter(c23,squeeze(f_origin),'same');%spm_erode(double(c23));
            sc23(mask<=0) = 0;
            sc1 = imfilter(c1,squeeze(f_origin),'same');
            sc1(mask<=0) = 0;
        elseif strcmpi(analysis,'FIX40')==1
            fMRIdatanii = load_untouch_nii([subdir,'\WM_fMRI\melodic.ica\filtered_func_data_clean.nii.gz']);
            fMRIdata = fMRIdatanii.img;
            mask = squeeze(fMRIdata(:,:,:,1))>0;
            fMRIdata = permute(fMRIdata,[4,1,2,3]);
            if FWHM>0
                fMRIdata = imfilter(fMRIdata,f_origin,'same');
                fMRIdata(:,mask<=0) = 0;
            end
            fMRIdata = fMRIdata(:,mask>0);
        elseif strcmpi(analysis,'confreg')==1 || strcmpi(analysis,'motreg')==1 || strcmpi(analysis,'WMCSF')==1
            load([subdir,'\WM_fMRI\s',num2str(FWHM),'tfMRI_WM_LR.mat'],'fMRIdata','mask');
            c2T1 = load_untouch_nii([subdir,'\T1\rc2T1w_restore_brain.nii']);
            sc2 = spm_erode(double(c2T1.img));
            while sum(sc2(:)>0.9)>20000
                sc2 = spm_erode(sc2);
            end
            c3T1 = load_untouch_nii([subdir,'\T1\rc3T1w_restore_brain.nii']);
            sc3 = spm_erode(double(c3T1.img));
            sc2 = sc2(mask>0)>0.9;sc3 = sc3(mask>0)>0.9;
            WM = mean(fMRIdata(:,sc2==1),2);
            CSF = mean(fMRIdata(:,sc3==1),2);
            [U,S,V]=svd(zscore(fMRIdata(:,sc2==1 | sc3==1)),'econ');
            compcor_PC = U(:,1:compcor_num);
            motion_reg = dlmread([subdir,'\WM_fMRI\Movement_Regressors_dt.txt']);
            motion_reg = motion_reg(16:end,:);
            if strcmpi(analysis,'confreg')==1
                confoundregressor = zscore([WM CSF compcor_PC motion_reg]);
            elseif strcmpi(analysis,'motreg')==1
                confoundregressor = zscore(motion_reg);
            elseif strcmpi(analysis,'WMCSF')==1
                confoundregressor = zscore([WM CSF compcor_PC]);
            end
            fMRIdata = function_RegressSignalOut(confoundregressor,fMRIdata,0);
        elseif strcmpi(analysis,'TF')==1
            load([subdir,'\WM_fMRI\s',num2str(FWHM),'tfMRI_WM_LR.mat'],'fMRIdata','mask');
            fMRIdata = function_detrend_LPF_motion(fMRIdata,0.72,1,1,1);
        end
        fMRIdata = ZY_detrend(fMRIdata,1,0);
        [Cor_tgtlure,Beta_tgtlure,Const_tgtlure,Res]=Univaranalysis(zscore(X_tgtlure(16:end,:)),...
            zscore(fMRIdata),mask,contrast_tgtlure,1);
        fMRIdata(:,isnan(Cor_tgtlure(mask>0))==1) = [];
        mask(isnan(Cor_tgtlure)==1) = 0;
        ZY_savenii(Cor_tgtlure,[analysisdir,'\s',num2str(FWHM),analysis,'_tgtlure_corr.nii.gz']);
%         ZY_savenii(Const_tgtlure,[analysisdir,'\s',num2str(FWHM),analysis,'_tgtlure_const.nii.gz']);
        
%         [Cor_block,Beta_block,Const_block,Res]=Univaranalysis(zscore(X_block(16:end,:)),...
%             zscore(fMRIdata),mask,contrast_block,1);
%         ZY_savenii(Cor_block,[analysisdir,'\s',num2str(FWHM),analysis,'_block_corr']);
%         ZY_savenii(Const_block,[analysisdir,'\s',num2str(FWHM),analysis,'_block_const.nii.gz']);
%         save([analysisdir,'\s',num2str(FWHM),analysis,'_result.mat'],'Cor_tgtlure','Cor_block','Beta_tgtlure','Beta_block',...
%             'Const_tgtlure','Const_block','X_tgtlure','X_block','contrast_tgtlure','contrast_block');
        if strcmpi(analysis,'GLM')==1
            save([subdir,'\WM_fMRI\s',num2str(FWHM),'tfMRI_WM_LR.mat'],'Cor_tgtlure','X_block','X_tgtlure',...
                'fMRIdata','mask','c1','sc1','c23','sc23','-v7.3');
        end
    end
end
