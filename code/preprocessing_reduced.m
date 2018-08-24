%% HDA-PROJECT - Preprocessing

clear; clc;

file.root = "..\OpportunityUCIDataset\dataset\";
file.dest = "data\reduced\";

params.num_subjects = 4;
params.num_sessions = 6;

index.features = [2:34 38:46 51:59 64:72 77:85 90:98 103:134];
index.labels = 244:250;

%% import sessions

for subject = 1:4
    disp("Importing data for subject " + int2str(subject))
    
    for session = 1:6
        
        % set filename with path
        if session < 6
            file.file = "S" + int2str(subject) + "-ADL" + int2str(session);
        else
            file.file = "S" + int2str(subject) + "-Drill";
        end
        file.name = file.root + file.file + ".dat";
        disp("Importing " + file.name)

        % load data and keep desired columns
        data = load(file.name);
        features = data(:,index.features);
        labels = data(:,index.labels);

        % cut head and tail of sessions, where ALL labels are 0
        % TRY ALSO WITH NaN in tail
        idx = zeros(2,7);
        for i = 1:7
            if numel(find(labels(:,i) ~= 0)) ~= 0
                idx(1,i) = find(labels(:,i) ~= 0, 1, 'first');
                idx(2,i) = find(labels(:,i) ~= 0, 1, 'last');
            else
                idx(1,i) = NaN;
                idx(2,i) = NaN;
            end
        end
        start = min(idx(1,:));
        stop = max(idx(2,:));
        disp("Cutting samples from "+int2str(start)+" to "+int2str(stop))
        features_cut = features(start:stop,:);
        labels_cut = labels(start:stop,:);  
        
        % interpolate with cubic splines
        temp_features = fillmissing(features_cut,'spline');

        % group features 3 by 3
        features_interp = zeros(size(temp_features,1),58);

        j = 1;
        for i=1:3:76
            features_interp(:,j) = sqrt(temp_features(:,i).^2 + temp_features(:,i+1).^2 + temp_features(:,i+2).^2);
%             features_interp(:,j) = (temp_features(:,i) + temp_features(:,i+1) + temp_features(:,i+2)) / 3;
            j = j + 1;
        end
        features_interp(:,27:end) = temp_features(:,79:end);
            
%             features_interp(:,13:18) = temp_features(:,37:42);
%             features_interp(:,19) = sqrt(temp_features(:,43).^2 + temp_features(:,44).^2 + temp_features(:,45).^2);
%             features_interp(:,20:25) = temp_features(:,46:51);
%             features_interp(:,26) = sqrt(temp_features(:,52).^2 + temp_features(:,53).^2 + temp_features(:,54).^2);
%             features_interp(:,27:32) = temp_features(:,55:60);
%             features_interp(:,33) = sqrt(temp_features(:,61).^2 + temp_features(:,62).^2 + temp_features(:,63).^2);
%             features_interp(:,34:39) = temp_features(:,64:69);
%             features_interp(:,40) = sqrt(temp_features(:,70).^2 + temp_features(:,71).^2 + temp_features(:,72).^2);
%             features_interp(:,41:end) = temp_features(:,73:end);
%             
        file.out = file.dest + file.file + ".mat";
        save(file.out, 'features_interp', 'labels_cut')
        disp("Stored at " + file.out)
        
       
    end
end
clear