clc
clear
close all
% measurenum=[30 24 18 12 6 29 23 17 11 5 28 22 16 10 4];


load ZigZagmove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat

% for i=1:5406
%     for j=1:15
%         andazeh1(i,j)=sqrt(sen1to30(i,3*measurenum(j)-2)^2+sen1to30(i,3*measurenum(j)-1)^2+sen1to30(i,3*measurenum(j))^2);
%     end
% end
for i=1:76866
    for j=1:2:30
        andazeh(j,i)=sqrt(sen1to30(i,3*(j)-2)^2+sen1to30(i,3*(j)-1)^2+sen1to30(i,3*(j))^2);
    end
end
andaze1=andazeh(any(andazeh,2),:);
andazeh1=andaze1(:,any(andaze1,1));

ecode=de2bi([1:2^(size(andazeh1,1))-1]);
for i=1:size(ecode,1)
    andazeh2=(ecode(i,:)'.*andazeh1);
    andazeh3{i} = andazeh2;
    andazeh3p{i}=i*ones(1,size(andazeh2,2));
end
andazeh4=cell2mat(andazeh3);
X=andazeh4;
T=repmat(magnetpos(1:2:76866,1:2),size(ecode,1),1)';
% net.output.processFcns = {'mapminmax'};

inputSize = 15;
numResponses  = 2;

layers = [ ...
    sequenceInputLayer(inputSize)
    fullyConnectedLayer(20)
    fullyConnectedLayer(numResponses )
    regressionLayer]



options = trainingOptions('adam', ...
    'MaxEpochs',10000, ...
    'MiniBatchSize',2, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',200,...
    'LearnRateDropFactor',0.9,...    
    'InitialLearnRate',0.1,...
    'Verbose',0);

net = trainNetwork(X,T,layers,options);


clear X T andazeh1 magnetpos andazeh2 andazeh3 andazeh3p andazeh3pp andazeh4 y sen1to30 andazeh


load Curvemove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat
T=magnetpos(:,1:2)';

for i=1:9566
    for j=1:2:30
        andazeh(i,j)=sqrt(sen1to30(i,3*(j)-2)^2+sen1to30(i,3*(j)-1)^2+sen1to30(i,3*(j))^2);
    end
end
andaze1=andazeh(any(andazeh,2),:);
andazeh1=andaze1(:,any(andaze1,1));

X=andazeh1';
y = predict(net,X);subplot(1,2,1)
plot(y(1,:))
hold on
plot(magnetpos(:,1))
subplot(1,2,2)
plot(-y(2,:))
hold on
plot(magnetpos(:,2))
SumationSqueredError=sum(sum([(y(1,:)-magnetpos(:,1)).^2;(-y(2,:)-magnetpos(:,2)).^2]));
maxerror=max(max([abs(y(1,:)-T(1,:));abs(-y(2,:)-T(2,:))]))

for i=1:9565
    for j=1:2
        smoothness(j,i)=(y(j,i)-y(j,i+1)).^2;
    end
end

totalsmoothness=sum(sum(smoothness));


figure

load Curvemove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat
T=magnetpos(:,1:2)';

for i=1:9566
    for j=1:230
        andazeh(i,j)=sqrt(sen1to30(i,3*(j)-2)^2+sen1to30(i,3*(j)-1)^2+sen1to30(i,3*(j))^2);
    end
end
andaze1=andazeh(any(andazeh,2),:);
andazeh1=andaze1(:,any(andaze1,1));

andazeh1(:,5)=zeros(9566,1);
X=andazeh1';
y = predict(net,X);subplot(1,2,1)
plot(y(1,:))
hold on
plot(magnetpos(:,1))
subplot(1,2,2)
plot(-y(2,:))
hold on
plot(magnetpos(:,2))
SumationSqueredError=sum(sum([(y(1,:)-magnetpos(:,1)).^2;(-y(2,:)-magnetpos(:,2)).^2]));
maxerror=max(max([abs(y(1,:)-T(1,:));abs(-y(2,:)-T(2,:))]))

for i=1:9565
    for j=1:2
        smoothness(j,i)=(y(j,i)-y(j,i+1)).^2;
    end
end

totalsmoothness=sum(sum(smoothness));
