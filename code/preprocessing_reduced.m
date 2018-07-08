%% HDA-PROJECT - Preprocessing

clear; clc;
root = "..\OpportunityUCIDataset\dataset\";
dest = "data\reduced\";
num_subjects = 4;
num_sessions = 6;
index_features = [2:46 51:59 64:72 77:85 90:98 103:134];
index_labels = 244:250;

%% import sessions

for subject = 1:4
    for session = 1:6
        if session < 6
            % set filename with path
            name = "S" + int2str(subject) + "-ADL" + int2str(session);
            ext = ".dat";
            filename = root + name + ext;
            disp("Importing " + filename)
            % load data and keep desired columns
            data = load(filename);
            features = data(:,index_features);
            labels = data(:,index_labels);
            % drop rows having label 0 for locomotion activity (column 1)
            features(labels(:,1)==0,:) = [];
            labels(labels(:,1)==0,:) = [];
            % interpolate with cubic splines
            filled_features = fillmissing(features,'spline');
            % group features 3 by 3
            reduced_features = zeros(size(filled_features,1),58);
            j = 1;
            for i=1:3:81-3
                reduced_features(:,j) = sqrt(filled_features(:,i).^2 + filled_features(:,i+1).^2 + filled_features(:,i+2).^2);
                j = j + 1;
            end
            reduced_features(:,j:end) = filled_features(:,82:end);
            % store to output
            output = dest + name + ".mat";
            save(output, 'reduced_features', 'labels')
            disp("Stored " + output)
        else
            % set filename with path
            name = "S" + int2str(subject) + "-Drill";
            ext = ".dat";
            filename = root + name + ext;
            disp("Importing " + filename)
            % load data and keep desired columns
            data = load(filename);
            features = data(:,index_features);
            labels = data(:,index_labels);
            % drop rows having label 0 for locomotion activity (column 1)
            features(labels(:,1)==0,:) = [];
            labels(labels(:,1)==0,:) = [];
            % interpolate with cubic splines
            filled_features = fillmissing(features,'spline');
            % group features 3 by 3
            reduced_features = zeros(size(filled_features,1),58);
            j = 1;
            for i=1:3:81-3
                reduced_features(:,j) = sqrt(filled_features(:,i).^2 + filled_features(:,i+1).^2 + filled_features(:,i+2).^2);
                j = j + 1;
            end
            reduced_features(:,j:end) = filled_features(:,82:end);
            % store to output
            output = dest + name + ".mat";
            save(output, 'reduced_features', 'labels')
            disp("Stored " + output)
        end
    end
end
clear