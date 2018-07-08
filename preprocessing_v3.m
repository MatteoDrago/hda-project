%% HDA-PROJECT - Preprocessing

clear; clc;
root = "..\OpportunityUCIDataset\dataset\";
num_subjects = 4;
num_sessions = 6;

%% import ADL sessions

for subject = 1:4
    for session = 1:5
        filename = root + "S" + int2str(subject) + "-ADL" + int2str(session) + ".dat";
        
        data = load(filename);
        features = data(:,[2:46 51:59 64:72 77:85 90:98 103:134]);
        labels = data(:,244:250);
        
        features(labels(:,1)==0,:) = [];
        labels(labels(:,1)==0,:) = [];
       
        filled_features = fillmissing(features,'spline');
        reduced_features = zeros(size(filled_features,1),58);
        j = 1;
        for i=1:3:81-3
            reduced_features(:,j) = sqrt(filled_features(:,i).^2 + filled_features(:,i+1).^2 + filled_features(:,i+2).^2);
            j = j + 1;
        end
        reduced_features(:,j:end) = filled_features(:,82:end);
        output = "prep\acc_magni\S" + int2str(subject) + "-ADL" + int2str(session) + ".mat";
        
        save(output, 'reduced_features', 'labels')
    end
end

%% import Drill sessions

for subject = 1:4
    filename = root + "S" + int2str(subject) + "-Drill.dat";
    data = load(filename);
    features = data(:,[2:46 51:59 64:72 77:85 90:98 103:134]);
    labels = data(:,[244:250]);
    
    features(labels(:,1)==0,:) = [];
    labels(labels(:,1)==0,:) = [];
    
    filled_features = fillmissing(features,'spline');
    reduced_features = zeros(size(filled_features,1),58);
    
    j = 1;
    for i=1:3:81-3
        reduced_features(:,j) = sqrt(filled_features(:,i).^2 + filled_features(:,i+1).^2 + filled_features(:,i+2).^2);
        j = j + 1;
    end
    
    output = "prep\acc_magni\S" + int2str(subject) + "-Drill.mat";
    
    save(output, 'reduced_features', 'labels')
end

