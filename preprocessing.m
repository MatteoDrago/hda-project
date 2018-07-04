%% HDA-PROJECT name = "S1-ADL1.dat";

clear; clc;
root = ".\OpportunityUCIDataset\dataset\";

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

        filled_features = fillmissing(features,'spline');

        idx = zeros(2,7);
        for i = 1:7
            idx(1,i) = find(labels(:,i) ~= 0, 1, 'first');
            idx(2,i) = find(labels(:,i) ~= 0, 1, 'last');
        end

        start = min(idx(1,:));
        stop = max(idx(2,:));

        filled_features = filled_features(start:stop,:);
        labels = labels(start:stop,:);

        output = "S" + int2str(subject);
        
        if session < 6
            output = output + "-ADL" + int2str(session) + ".mat";
        else
            output = output + "-Drill" + ".mat";
        end
        
        save(output, 'features', 'labels')
        
    end
end


%% CHECK NaNs
m = mean(filled_features);
