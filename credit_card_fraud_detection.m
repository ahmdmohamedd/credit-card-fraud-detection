% Load Data
data = readtable('creditcard.csv'); % Assuming the dataset is named 'creditcard.csv'

% Preprocessing
% Extract features and labels
X = table2array(data(:, 1:end-1)); % Features
y = table2array(data(:, end)); % Labels (fraud: 1, non-fraud: 0)

% Feature scaling (normalize 'Time' and 'Amount' columns)
X(:,1) = (X(:,1) - mean(X(:,1))) / std(X(:,1)); % Normalize 'Time'
X(:,end) = (X(:,end) - mean(X(:,end))) / std(X(:,end)); % Normalize 'Amount'
X(:,2:end-1) = (X(:,2:end-1) - mean(X(:,2:end-1))) ./ std(X(:,2:end-1)); % Normalize other features

% Hybrid Sampling
% Separate fraud and non-fraud data
fraud_data = X(y == 1, :);
non_fraud_data = X(y == 0, :);

% Oversample fraud data to match half the size of non-fraud data
num_oversamples = floor(size(non_fraud_data, 1) / 2);
oversampled_fraud_data = fraud_data(randi(size(fraud_data, 1), num_oversamples, 1), :);

% Combine oversampled fraud data with original fraud and undersampled non-fraud data
undersampled_non_fraud_data = non_fraud_data(randperm(size(non_fraud_data, 1), num_oversamples), :);
X_balanced = [oversampled_fraud_data; fraud_data; undersampled_non_fraud_data];
y_balanced = [ones(num_oversamples, 1); ones(size(fraud_data, 1), 1); zeros(num_oversamples, 1)];

% Add a column of ones to X_balanced for the intercept term
[m, n] = size(X_balanced);
X_balanced = [ones(m, 1), X_balanced];

% Initialize Parameters
theta = zeros(n + 1, 1); % Including the intercept term
alpha = 0.01; % Learning rate
num_iterations = 1000; % Number of gradient descent steps
fraud_weight = 10; % Adjust based on experimentation

% Sigmoid Function
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

% Weighted Cost Function
function J = computeCostWeighted(X, y, theta, fraud_weight)
    m = length(y);
    h = sigmoid(X * theta);
    
    % Assign higher weight to fraud cases
    weights = ones(m, 1);
    weights(y == 1) = fraud_weight;
    
    J = (1 / m) * sum(weights .* (-y .* log(h) - (1 - y) .* log(1 - h)));
end

% Weighted Gradient Descent
function [theta, J_history] = gradientDescentWeighted(X, y, theta, alpha, num_iterations, fraud_weight)
    m = length(y);
    J_history = zeros(num_iterations, 1);

    for iter = 1:num_iterations
        h = sigmoid(X * theta);
        error = h - y;
        
        % Adjust gradient by fraud_weight for fraud cases
        weights = ones(m, 1);
        weights(y == 1) = fraud_weight;
        gradient = (1 / m) * (X' * (weights .* error));
        
        theta = theta - alpha * gradient;

        % Save the weighted cost J at each iteration
        J_history(iter) = computeCostWeighted(X, y, theta, fraud_weight);
    end
end

% Train Model with Weighted Gradient Descent
[theta, J_history] = gradientDescentWeighted(X_balanced, y_balanced, theta, alpha, num_iterations, fraud_weight);

% Plot Cost Function (Model Convergence)
figure;
plot(1:num_iterations, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
title('Convergence of Weighted Gradient Descent');

% Prediction Function with Threshold Adjustment
function p = predictWithThreshold(X, theta, threshold)
    p = sigmoid(X * theta) >= threshold;
end

%% Set a lower threshold for fraud sensitivity
threshold = 0.6;
predictions = predictWithThreshold(X_balanced, theta, threshold);

% Model Evaluation
% Confusion Matrix
tp = sum((predictions == 1) & (y_balanced == 1));
tn = sum((predictions == 0) & (y_balanced == 0));
fp = sum((predictions == 1) & (y_balanced == 0));
fn = sum((predictions == 0) & (y_balanced == 1));

% Accuracy, Precision, Recall, F1-Score
accuracy = (tp + tn) / m;
precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1-Score: %.2f\n', f1_score);

%% Plot Confusion Matrix
confusion_matrix = [tp, fp; fn, tn];
figure;
heatmap({'Fraud', 'Non-Fraud'}, {'Fraud', 'Non-Fraud'}, confusion_matrix, 'Colormap', parula, 'ColorbarVisible', 'on');
xlabel('Predicted Class');
ylabel('Actual Class');
title('Confusion Matrix');

% Visualize Class Distribution
figure;
labels = categorical(y_balanced, [0 1], {'Non-Fraud', 'Fraud'});
histogram(labels);
xlabel('Class');
ylabel('Frequency');
title('Class Distribution in Balanced Dataset');

% Visualize Fraud and Non-Fraud Transactions (PCA Features)
figure;
gscatter(X_balanced(:,2), X_balanced(:,3), y_balanced, 'rb', 'xo');
xlabel('Feature V1');
ylabel('Feature V2');
title('Fraud vs Non-Fraud Transactions');
legend({'Non-Fraud', 'Fraud'});
