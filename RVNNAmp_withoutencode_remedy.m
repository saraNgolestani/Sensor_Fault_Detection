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
for i=1:2:76866
    for j=1:4:30
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
net.output.processFcns = {'mapminmax'};
net = feedforwardnet([15 10 10]);%layrecnet(5,2);%feedforwardnet([7 3]);%feedforwardnet([10 6]);%[15 10 7]
net.trainFcn = 'trainbr';%trainscg trainrp trainoss
% net.trainParam.lr=0.2;%0.2
net.trainParam.max_fail=60;%20
% net.trainParam.mc=0.95;%0.05
net.trainParam.epochs=5000;%2000
net = train(net,X,T,'useGPU','no');
% save([num2str(i)])


clear X T andazeh1 magnetpos andazeh2 andazeh3 andazeh3p andazeh3pp andazeh4 y sen1to30 andazeh
load Curvemove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat
for i=1:9566
    for j=1:4:30
        andazeh(j,i)=sqrt(sen1to30(i,3*(j)-2)^2+sen1to30(i,3*(j)-1)^2+sen1to30(i,3*(j))^2);
    end
end
andaze1=andazeh(any(andazeh,2),:);
andazeh1=andaze1(:,any(andaze1,1));
andazeh1(5,:)=zeros(1,9566);
X=(andazeh1);

T=magnetpos(:,1:2)';

y =net(X);
y = smoothdata(y,2,'movmean',[200,0]);
subplot(4,2,1)
plot(y(1,:))
hold on
plot(T(1,:))
legend('Estimated','Real')
xlabel('Samples')
ylabel('Position (X axis)')
title('Neural Network remedy 15 sensor')

subplot(4,2,3)
plot(-y(2,:))
hold on
plot(T(2,:))

legend('Estimated','Real')
xlabel('Samples')
ylabel('Position (Y axis)')

subplot(4,2,5)
plot(y(1,:),-y(2,:))
hold on
plot(T(1,:),T(2,:))
legend('Estimated','Real')
xlabel('Position (X axis)')
ylabel('Position (Y axis)')


subplot(4,2,7)
plot(sum([(y(1,:)-T(1,:)).^2;(-y(2,:)-T(2,:)).^2]))
xlabel('Samples')
ylabel('Squered Error')
title(['Sumation of Squered Error is ',num2str(sum(sum([(y(1,:)-T(1,:)).^2;(-y(2,:)-T(2,:)).^2])))])

error1=sum(sum([(y(1,:)-T(1,:)).^2;(-y(2,:)-T(2,:)).^2]));
maxerror=max(max([abs(y(1,:)-T(1,:));abs(-y(2,:)-T(2,:))]))




