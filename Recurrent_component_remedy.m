clc
clear
close all
% measurenum=[30 24 18 12 6 29 23 17 11 5 28 22 16 10 4];


load ZigZagmove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat
%%Selecting Sensors 
Amp=[];
for j=1:4:30
    Amp=[Amp sen1to30(1:2:76866,3*(j)-2) sen1to30(1:2:76866,3*(j)-1) sen1to30(1:2:76866,3*(j))];
end
%% Making Rich dataset
Amp1=Amp(any(Amp,2),:);
Amplitude1=(Amp1(:,any(Amp1,1)))';
clear Amp Amp1
ecode=de2bi([1:2^(size(Amplitude1,1)/3-rem(size(Amplitude1,1),3))-1]);
for i=1:size(ecode,1)
    for j=1:size(Amplitude1,1)/3-rem(size(Amplitude1,1),3)
        Amp2(3*j-2:3*j,:)=(ecode(i,j)'.*Amplitude1(3*j-2:3*j,:));
    end
    Amp3{i} = Amp2;
    Amp3P{i}=i*ones(1,size(Amp2,2));
end
Amp4=cell2mat(Amp3);
X=Amp4;
T=repmat(magnetpos(1:2:76866,1:2),size(ecode,1),1)';
net.output.processFcns = {'mapminmax'};
net = layrecnet(1,[20 10 5]);%layrecnet(5,2);%feedforwardnet([7 3]);%feedforwardnet([10 6]);%[15 10 7]
net.trainFcn = 'trainbr';%trainscg trainrp trainoss
% net.trainParam.lr=0.2;%0.2
net.trainParam.max_fail=60;%20
% net.trainParam.mc=0.95;%0.05
net.trainParam.epochs=5000;%2000
net = train(net,X,T,'useGPU','no');
% save([num2str(i)])


%%Test begins

clear X T Amplitude magnetpos Amp2 Amp3 Amp3P Amp3PP Amp4 y sen1to30 Amp
load Curvemove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat
Amp=[];
for j=1:4:30
    Amp=[Amp sen1to30(1:9566,3*(j)-2) sen1to30(1:9566,3*(j)-1) sen1to30(1:9566,3*(j))];
end

Amp1=Amp(any(Amp,2),:);
Amplitude1=(Amp1(:,any(Amp1,1)))';
Amplitude1(3*5-2:3*5,:)=zeros(3,9566);
X=(Amplitude1);

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




