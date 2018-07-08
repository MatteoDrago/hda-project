%% HDA-PROJECT

clear; clc;
base = "../OpportunityUCIDataset/dataset/";

for subject = 1:4
    
    % training set
    sadl1 = load(base+"S1-ADL1.dat");
    sadl2 = load([base 'S1-ADL2.dat']);
    sadl3 = load([base 'S1-ADL3.dat']);
    sdrill = load([base 'S1-Drill.dat']);
    
    % test set
    sadl4 = load([base 'S1-ADL4.dat']);
    sadl5 = load([base 'S1-ADL5.dat']);
    
    
end



%% import ADL sessions

for subject = 1:4
    for session = 1:5
        filename = root + "S" + int2str(subject);
        if session < 6
            filename = filename + "-ADL" + int2str(session) + ".dat";
        else
            filename = filename + "-Drill" + ".dat";
        end
        
        data = load(filename);
        features = data(:,[2:46 51:59 64:72 77:85 90:98 103:134]);
        labels = data(:,[244:250]);

        plot(data(:,1), labels(:,1))
        
        idx = zeros(2,7);
        for i = 1:7
            idx(1,i) = find(labels(:,i) ~= 0, 1, 'first');
            idx(2,i) = find(labels(:,i) ~= 0, 1, 'last');
        end

        start = min(idx(1,:));
        stop = max(idx(2,:));

        features = features(start:stop,:);
        labels = labels(start:stop,:);
        filled_features = fillmissing(features,'spline');

        plot(1:1:length(features(:,1)), normalize(filled_features))

        output = "data_temp\S" + int2str(subject);
        
        if session < 6
            output = output + "-ADL" + int2str(session) + ".mat";
        else
            output = output + "-Drill" + ".mat";
        end
        
        save(output, 'filled_features', 'labels')
        
    end
end

%% import Drill sessions

for subject = 1:4
    filename = root + "S" + int2str(subject) + "-Drill.dat";
    data = load(filename);
    features = data(:,[2:46 51:59 64:72 77:85 90:98 103:134]);
    labels = data(:,[244:250]);
    filled_features = fillmissing(features,'spline');
    output = "data_temp\S" + int2str(subject) + "-Drill.mat";
    %save(output, 'filled_features', 'labels')
end

%% CHECK NaNs
m = isnan(filled_features);
ms = sum(m)
find(ms)

%% CLASSIFICATION


