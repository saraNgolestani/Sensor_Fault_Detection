clc
clear
close all
% measurenum=[30 24 18 12 6 29 23 17 11 5 28 22 16 10 4];


load ZigZagmove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat

andazeh=[];
    for j=1:4:30
        andazeh=[andazeh sen1to30(1:2:76866,3*(j)-2) sen1to30(1:2:76866,3*(j)-1) sen1to30(1:2:76866,3*(j))];
    end

andaze1=andazeh(any(andazeh,2),:);
andazeh1=(andaze1(:,any(andaze1,1)))';

ecode=de2bi([1:2^(size(andazeh1,1)/3-rem(size(andazeh1,1),3))-1]);
for i=1:size(ecode,1)
    for j=1:size(andazeh1,1)/3-rem(size(andazeh1,1),3)
        andazeh2(3*j-2:3*j,:)=(ecode(i,j)'.*andazeh1(3*j-2:3*j,:));
    end
    andazeh3{i} = andazeh2;
    andazeh3p{i}=i*ones(1,size(andazeh2,2));
end
andazeh4=cell2mat(andazeh3);
X=andazeh4;
T=repmat(magnetpos(1:2:76866,1:2),size(ecode,1),1)';
net.output.processFcns = {'mapminmax'};
net = layrecnet(30);%layrecnet(5,2);%feedforwardnet([7 3]);%feedforwardnet([10 6]);%[15 10 7]
net.trainFcn = 'trainbr';%trainscg trainrp trainoss
% net.trainParam.lr=0.2;%0.2
net.trainParam.max_fail=60;%20
% net.trainParam.mc=0.95;%0.05
net.trainParam.epochs=8000;%2000%5000
net = train(net,X,T,'useGPU','no');
% save([num2str(i)])


clear X T andazeh1 magnetpos andazeh2 andazeh3 andazeh3p andazeh3pp andazeh4 y sen1to30 andazeh
load Curvemove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat
andazeh=[];
    for j=1:4:30
        andazeh=[andazeh sen1to30(1:9566,3*(j)-2) sen1to30(1:9566,3*(j)-1) sen1to30(1:9566,3*(j))];
    end

andaze1=andazeh(any(andazeh,2),:);
andazeh1=(andaze1(:,any(andaze1,1)))';
andazeh1(3*5-2:3*5,:)=zeros(3,9566);
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




