%% HDA-PROJECT - Preprocessing

% Starting from the original dataset, this routine preserve all features
% cutting out head and tail, given that we assume them as useless
% transient.

% pick up data 
clear; clc;

file.root = "..\OpportunityUCIDataset\dataset\";
file.dest = "data\";

params.num_subjects = 4;
params.num_sessions = 6;

index.features = [2:34 38:46 51:59 64:72 77:85 90:98 103:134];
index.labels = 244:250;

%% import sessions

% subject=1;session=1;
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
        features_temp = data(:,index.features);
        labels_temp = data(:,index.labels);
        
        % cut head and tail of sessions, where ALL labels are 0
        % TRY ALSO WITH NaN in tail
        idx = zeros(2,7);
        for i = 1:7
            if numel(find(labels_temp(:,i) ~= 0)) ~= 0
                idx(1,i) = find(labels_temp(:,i) ~= 0, 1, 'first');
                idx(2,i) = find(labels_temp(:,i) ~= 0, 1, 'last');
            else
                idx(1,i) = NaN;
                idx(2,i) = NaN;
            end
        end
        start = min(idx(1,:));
        stop = max(idx(2,:));
        disp("Cutting samples from "+int2str(start)+" to "+int2str(stop))
        features = features_temp(start:stop,:);
        labels = labels_temp(start:stop,:);

        % interpolate with cubic splines
        missing = sum(sum(isnan(features)));
        disp("Interpolating " + int2str(missing) + " NaN values")
        features = fillmissing(features,'spline');
        missing = sum(sum(isnan(features)));

        file.out = file.dest + file.file + ".mat";
        save(file.out, 'features', 'labels')
        disp("Stored at " + file.out)
    end
end
clear