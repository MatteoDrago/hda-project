%% HDA-PROJECT

clear; clc;
root = ".\OpportunityUCIDataset\dataset\";

%% import ADL sessions

data = load(".\OpportunityUCIDataset\dataset\S1-ADL5.dat");
features = data(:,[2:46 51:59 64:72 77:85 90:98 103:134]);
labels = data(:,[244:250]);

temp = isnan(features(:,1));
H = zeros(size(temp));
count = 0;
for idx = 1:numel(temp)
  if temp(idx)
    count = count + 1;
  else
    count = 0;
  end
  H(idx) = count;
end


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
       


