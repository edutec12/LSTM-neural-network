clc;
clear;

data = [1106.85 1087.4 1108.72 1113.81 1097.37 1104.47 1087.98 1100.45 1085.71 1260.74 1399.54 1401.64 1364.16 1317.99 1257.02 1229.45 1208.16 1197.39 1171.28 1142.56 1118.88 1101.13 1071.33 1124.27 1167.43 1162.01 1128.23 1129.54 1103.56 1028.29 1022.48 976.95 949.84 1063.7 1201.21 1181.28 1186.68 1111.07 1070.54 1048.08 1021.49 1022.36 1019.62 982.89 972.95 957.25 930.14 1005.13 1016.61 1035.51 1053.27 1090.26 1078.38 1084.6 1060.79 1020.59 980.72 1119.97 1245.12 1245.33 1172.2 1159.39 1109.87 1072.5 1037.9 1027.98 1044.49 1009.5 1009.28 999.93 986.31 981.29 1000.42 1009.31 1060 1103.37 1096.64 1081.34 1077.82 1051.05 989.7 1103.54 1268.18 1208.2 1203.75 1171.06 1130 1075.27 1051.94 1063.83 1029.4 981.92 980.64 981.14 943.14 1012.9 978.18 976.84 1049.77 1099.9 1070.1 1055.66 1055.87 1015.34 984.54 1133.81 1262.55 1223.35 1211.25 1173.11 1143.66 1127.64 1075.4 1037.72 1036.39 1010.3 1005.77 1005.12 992.84 1003.7 1016.4 1020.19 1089.76 1105.86 1040.65 1027.47 1048.93 997.46 964.23 1086.14 1209.82 1190.69 1166.56 1123.74 1063.3 1030.28 1023.55];
numTimeStepsTrain = numel(data)-1;
train_data = data(1:end-1);
test_data = data(end);

mu = mean(train_data);
sig = std(train_data);
train_data = (train_data - mu) / sig;
test_data = (test_data - mu) / sig;
XTrain = train_data(1:end-1);
YTrain = train_data(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;
layers = [sequenceInputLayer(numFeatures) lstmLayer(numHiddenUnits) fullyConnectedLayer(numResponses) regressionLayer];
options = trainingOptions('adam', 'MaxEpochs',100, 'GradientThreshold',1, 'InitialLearnRate',0.01, 'LearnRateSchedule','piecewise', 'LearnRateDropFactor',0.2, 'LearnRateDropPeriod',50, 'Verbose',0);
net = trainNetwork(XTrain,YTrain,layers,options);

numTimeStepsTest = 20;
XTest = test_data;
net = predictAndUpdateState(net,XTrain);
YPred = zeros(numTimeStepsTest, 1);

for i = 1:numTimeStepsTest
[net,YPred(i)] = predictAndUpdateState(net,XTest,'ExecutionEnvironment','cpu');
XTest = YPred(i);
end

YPred = sig*YPred + mu;
figure
plot(data,'.-')
xlabel('Time')
ylabel('Data')
title('Forecast')
legend('Observed','Forecast')
hold on
plot(numTimeStepsTrain+2:numTimeStepsTrain+numTimeStepsTest+1,YPred,'.-')
xlabel('Time')
ylabel('Data')
title('Forecast')
legend('Observed','Forecast')
display([data, YPred.'])


