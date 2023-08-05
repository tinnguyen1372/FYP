clear all
close all
clc

N_trunk = 1000;

%set parameters
Rmin_trunk = 0.15;
Rmax_trunk = 0.30;
CenTrunk_X = 0;
CenTrunk_Y = 0;
N_angle = 360;
safety_d = 0.01; % safe distance(Rmin_trunk_radius - max_cavity_radius - safety_d)

%small cavity, if no need can remove
Rmin_Scavity = 0.03;
Rmax_Scavity = 0.09;

layer1 = [1, 1, 0];
layer2 = [1, 0.8, 0];
layer3 = [1, 0.6, 0 ];
%layer4 = [1, 0.4, 0];
cavity_colour = [1, 0.2, 0];
rgbTgray = [0.2989;0.5870;0.1140];

layer1_G = layer1*rgbTgray;
layer2_G = layer2*rgbTgray;
layer3_G = layer3*rgbTgray;
%layer4_G = layer4*rgbTgray;
layer_hole = cavity_colour*rgbTgray;



x_trunk_L1 = zeros(N_trunk,N_angle);
y_trunk_L1 = zeros(N_trunk,N_angle);

x_trunk_L2 = zeros(N_trunk,N_angle);
y_trunk_L2 = zeros(N_trunk,N_angle);

x_trunk_L3 = zeros(N_trunk,N_angle);
y_trunk_L3 = zeros(N_trunk,N_angle);

x_Bcavity = zeros(N_trunk,N_angle);
y_Bcavity = zeros(N_trunk,N_angle);

r_trunk_L3 = zeros(N_trunk,N_angle);
r_trunk_L2 = zeros(N_trunk,N_angle);
r_trunk_L1 = zeros(N_trunk,N_angle);
r_Bcavity = zeros(N_trunk,N_angle);

hole_factor = zeros(N_trunk,1);
R_center_Bcavity=zeros(N_trunk,1);
theta_Bcavity=zeros(N_trunk,1);
CenBcavity_X=zeros(N_trunk,1);
CenBcavity_Y=zeros(N_trunk,1);

n_fig =1;
%up: upperbound, IS: conductive layer, HW: heartwood
up_IS = 0.9;
lo_IS = 0.75;
% up_HW =0.4;
% lo_HW=0.2;
up_bark = 1.1;
lo_bark = 1.05;


for i = 1:N_trunk

    %Randomize amplitude and phase.
    H = 8;
    R_rand_trunk = rand(1,1)*(Rmax_trunk-Rmin_trunk)+Rmin_trunk;
    [x_trunk_L2(i,:),y_trunk_L2(i,:),r_trunk_L2(i,:)]=EnclosedCurve(R_rand_trunk,N_angle,13,CenTrunk_X,CenTrunk_Y);
    %     [x_trunk_L1(i,:),y_trunk_L1(i,:),r_trunk_L1(i,:)]=bark(r_trunk_L2(i,:),25,CenTrunk_X,CenTrunk_Y);

    ratio_L3 = rand*(up_IS-lo_IS)+lo_IS;
    x_trunk_L3(i,:)=ratio_L3*x_trunk_L2(i,:);
    y_trunk_L3(i,:)=ratio_L3*y_trunk_L2(i,:);
    r_trunk_L3(i,:)=ratio_L3*r_trunk_L2(i,:);

    ratio_L1 = rand*(up_bark-lo_bark)+lo_bark;
    x_trunk_L1(i,:) = ratio_L1*x_trunk_L2(i,:);
    y_trunk_L1(i,:) = ratio_L1*y_trunk_L2(i,:);
    r_trunk_L1(i,:) = ratio_L1*r_trunk_L2(i,:);

    %     ratio_L4 =rand*(up_HW-lo_HW)+lo_HW;
    %     x_trunk_L4(i,:)=ratio_L4*x_trunk_L2(i,:);
    %     y_trunk_L4(i,:)=ratio_L4*y_trunk_L2(i,:);
    %     r_trunk_L4(i,:)=ratio_L4*r_trunk_L2(i,:);


    Rmin_trunk_L3 = min(r_trunk_L3(i,:));
    Rmax_Bcavity = 0.8*Rmin_trunk_L3;
    Rmin_Bcavity = 0.05;
    hole_factor(i)=rand(1,1);
    R_rand_Bcavity = hole_factor(i)*(Rmax_Bcavity-Rmin_Bcavity)+Rmin_Bcavity;

    min_d_Bcavity = 0.02;
    max_d_Bcavity = Rmin_trunk_L3 - R_rand_Bcavity - safety_d;
    R_center_Bcavity(i) = min_d_Bcavity + (max_d_Bcavity-min_d_Bcavity) .* rand;
    theta_Bcavity(i) = 2*pi*rand; % compute normally distributed values with zero mean and standard deviation of 2*pi
    %theta_Bcavity = mod(thetaNormal_Bcavity,2*pi); % map them to the range [0,2*pi]
    CenBcavity_X(i) = R_center_Bcavity(i).*cos(theta_Bcavity(i));
    CenBcavity_Y(i) = R_center_Bcavity(i).*sin(theta_Bcavity(i));


    shape_factor(i) = rand(1,1);
    max_shape = 20;
    min_shape = 15;
    shape_cavity(i) = round(shape_factor(i)*(max_shape-min_shape)+min_shape);
    [x_Bcavity(i,:),y_Bcavity(i,:),r_Bcavity(i,:)]=EnclosedCurve(R_rand_Bcavity,N_angle,shape_cavity(i),CenBcavity_X(i),CenBcavity_Y(i));
    

    %round the parameters to format with only 3 unit of decimal.


    %     range_trunk is the xmin,xmax,ymin,ymax of a single trunk, this is used to
    %     determine the position of the TxRx in the simulation file.
    range_trunk(:,1)= min(x_trunk_L1,[],2);
    range_trunk(:,2)= max(x_trunk_L1,[],2);
    range_trunk(:,3)= min(y_trunk_L1,[],2);
    range_trunk(:,4)= max(y_trunk_L1,[],2);

    %Plot x(t) and y(t), fill -> fill colour with what u want
    figure (1);
    fill(x_trunk_L1(i,:),y_trunk_L1(i,:),layer1,'LineStyle','none');
    hold on
    fill(x_trunk_L2(i,:),y_trunk_L2(i,:),layer2,'LineStyle','none');
    hold on
    fill(x_trunk_L3(i,:),y_trunk_L3(i,:),layer3,'LineStyle','none');
    hold on
    fill(x_Bcavity(i,:),y_Bcavity(i,:),cavity_colour,'LineStyle','none');
    xlabel('x(t)');
    ylabel('y(t)');
    set(gcf,'position',[0,0,1000,1000]);%320=400
    axis([-0.35 0.35 -0.35 0.35]);
    %axis equal;
    axis off;
    set(gcf,'visible','off');
    set(gcf,'PaperPositionMode','auto');
    %img_name = strcat('minor_decay/ML_minor_decay_',num2str(i),'.png');
    %img_name1 = strcat('../image/decay/decay_',num2str(i),'.png');
    img_name2 = strcat('../image/defect/defect',num2str(i),'.png');
    %     print('-dpng','-r0',img_name1);
    %     print('-dpng','-r0',img_name2);
    h=getframe(gca);
    %imwrite(h.cdata,img_name1,'png');
    imwrite(h.cdata,img_name2,'png');
    hold off;
    %saveas(gca,['BigCavity',num2str(i),'.jpg']);
    %n_fig = n_fig+1;

    close;


    figure (3);
    fill(x_trunk_L1(i,:),y_trunk_L1(i,:),layer1,'LineStyle','none');
    hold on
    fill(x_trunk_L2(i,:),y_trunk_L2(i,:),layer2,'LineStyle','none');
    hold on
    fill(x_trunk_L3(i,:),y_trunk_L3(i,:),layer3,'LineStyle','none');
    xlabel('x(t)');
    ylabel('y(t)');
    set(gcf,'position',[0,0,1000,1000]);%320=400
    axis([-0.35 0.35 -0.35 0.35]);
    %axis equal;
    axis off;
    set(gcf,'visible','off');
    set(gcf,'PaperPositionMode','auto');

    img_name1 = strcat('../image/healthy/healthy',num2str(i),'.png');

    h=getframe(gca);
    imwrite(h.cdata,img_name1,'png');
    hold off;
    close;

end
hole_flag = hole_factor;
hole_flag(hole_flag<=0.33)=-1;
hole_flag((hole_flag>0.33)&(hole_flag<0.66))=0;
hole_flag(hole_flag>=0.66)=1;
height_factor = rand(1,1000);

save('SL_trunk.mat','x_trunk_L1','y_trunk_L1','x_trunk_L2','y_trunk_L2',...
    'x_trunk_L3','y_trunk_L3','x_Bcavity','y_Bcavity','range_trunk',...
    'hole_factor','hole_flag','r_trunk_L1','r_trunk_L2','r_trunk_L3',...
    'height_factor','R_center_Bcavity','theta_Bcavity','r_Bcavity',...
    'CenBcavity_X','CenBcavity_Y','CenTrunk_X','CenTrunk_Y');