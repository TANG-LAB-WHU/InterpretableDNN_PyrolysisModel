
trainData = input;
% Look for the linear indices of elements in the trainData
linear_nan_indices = find(isnan(trainData));

% break down linear_nan_indices into row and column indices
[nan_rows, nan_cols] = ind2sub(size(trainData), linear_nan_indices);

% nan_rows record the row number having NaN.
% nan_cols record the column having NaN.

% Showcase the location of NaN：
for i = 1:length(nan_rows)
    nan_value = trainData(nan_rows(i), nan_cols(i));
    disp(['NaN locates ', num2str(nan_rows(i)), ' row，and ', num2str(nan_cols(i)), ' column.']);
end

% or directlyy access the datapoint having NaN：
%nan_elements = trainData(isnan(trainData));