function PhiUnwrap = MCF(Psi, options)
% 算法: Minimum Network Flow
%
% INPUTS:
%   - Psi: 2维缠绕相位(rad)
% OUTPUT:
%   - Phi: 2维解缠相位(rad)
%% 输入参数
if nargin<2
    options = struct();
end
if ndims(Psi)>2
    error('CUNWRAP: input Psi must be 2D array.');
end
[ny nx] = size(Psi);
if nx<2 || ny<2
    error('CUNWRAP: size of Psi must be larger than 2');
end
%残差值控制
roundK = getoption(options, 'roundk', false);
% 默认权重(Default weight)
w1 = ones(ny,1); w1([1 end]) = 0.5;
w2 = ones(1,nx); w2([1 end]) = 0.5;
weight = w1*w2; % tensorial product张量积
weight = getoption(options, 'weight', weight);
% 分块
% 不分快结果较好(?)-------------20150415
bs = max(ny, nx);
blocksize = getoption(options, 'maxblocksize', bs);%maxblocksize是子块大小
blocksize = max(blocksize,2);
% 块与块之间重叠部分(25%)
p = getoption(options, 'overlap', 0.25);
p = max(min(p,1),0);
% 子块索引
[ix blk_x] = splitidx(blocksize, nx, p);%blk_x 是列方向子块尺寸（像素个数），ix是列方向子块下标序列集
[iy blk_y] = splitidx(blocksize, ny, p);%blk_y 是行方向子块尺寸（像素个数），iy是行方向子块下标序列集
%子块个数
nbblk = length(iy)*length(ix);
% %% write parameters
% fid1 = fopen('.\Mid\MCF_unwrap_parameters.txt','wt');
% fprintf(fid1,'==========缠绕干涉相位信息=========\n');
% fprintf(fid1,'%s %i \n','干涉相位尺寸(行): ',ny);
% fprintf(fid1,'%s %i \n','干涉相位尺寸(列): ',nx);
% fprintf(fid1,'==========MCF解缠处理信息==========\n');
% fprintf(fid1,'%s %i \n','子块个数:     ',nbblk);
% fprintf(fid1,'%s %i \n','子块尺寸(行): ',blk_y);
% fprintf(fid1,'%s %i \n','子块尺寸(列): ',blk_x);
% fprintf(fid1,'%s %i %s \n','子块重叠率:   ',p*100,'%');
% fclose(fid1);
%% 分配阵列、缠绕相位偏导的正＼负部分
% negative/positive parts of wrapped partial derivatives　
x1p = nan(ny-1, nx, class(Psi));%nan,;class:返回参数对象的类型名称
x1m = nan(ny-1, nx, class(Psi));
x2p = nan(ny, nx-1, class(Psi));
x2m = nan(ny, nx-1, class(Psi));
%% 子块数控制的循环，分块解缠
blknum = 0;
for i=1:length(iy)%length(iy)行子块个数
    iy0 = iy{i};%各行子块起点终点集
    iy1 = iy0(1:end-1);
    iy2 = iy0(1:end);    
    for j=1:length(ix)%length(ix)列子块个数
        ix0 = ix{j};%各列子块起点终点集
        ix1 = ix0(1:end);
        ix2 = ix0(1:end-1);       
        blknum = blknum + 1;
        options.weight = weight(iy0,ix0);%输入权重矩阵
        options.blknum = blknum;         %输入子块序号
        % Psi(iy0,ix0)是分块对应的缠绕相位,RELAX策略求解线性最小化问题
        % cunwrap_blk返回最小化流量x1p,x1m,x2p,x2m
        % 通过x1p,x1m,x2p,x2m计算k1,k2估计残差
        [x1p(iy1,ix1) x1m(iy1,ix1) x2p(iy2,ix2) x2m(iy2,ix2)] = ...
            cunwrap_blk(Psi(iy0,ix0), ...
            x1p(iy1,ix1), x1m(iy1,ix1),...
            x2p(iy2,ix2), x2m(iy2,ix2),...
            options);
    end
end
%% 计算行方向（y-direction）偏导数
% Compute partial derivative Psi1, eqt (1,3)
i = 1:(ny-1);
j = 1:nx;
[ROW_I ROW_J] = ndgrid(i,j);
I_J = sub2ind(size(Psi),ROW_I,ROW_J);
IP1_J = sub2ind(size(Psi),ROW_I+1,ROW_J);
Psi1 = Psi(IP1_J) - Psi(I_J);
Psi1 = mod(Psi1+pi,2*pi)-pi;%Psi1 is in [-pi,pi)
%% 计算列方向（x-direction）偏导数
% Compute partial derivative Psi2, eqt (2,4)
i = 1:ny;
j = 1:(nx-1);
[ROW_I ROW_J] = ndgrid(i,j);
I_J = sub2ind(size(Psi),ROW_I,ROW_J);
I_JP1 = sub2ind(size(Psi),ROW_I,ROW_J+1);
Psi2 = Psi(I_JP1) - Psi(I_J);
Psi2 = mod(Psi2+pi,2*pi)-pi;%Psi2 is in [-pi,pi)
%% 计算偏导数跳变点, eqt (20,21)
% 通过x1p,x1m,x2p,x2m计算最小偏导残差k1,k2
k1 = x1p-x1m;
k2 = x2p-x2m;
% Round to integer solution (?)
% 整数化k1,k2
if roundK
    k1 = round(k1);
    k2 = round(k2);
end
%% 偏导数跳变点求和
% Sum the jumps with the wrapped partial derivatives, eqt (10,11)
% 计算解缠相位(真实相位)偏导(梯度),即delta1_Phi(i,j)，delta2_Phi(i,j)
k1 = reshape(k1,[ny-1 nx]);
k2 = reshape(k2,[ny nx-1]);
%k1是delta1_Phi(i,j)/2pi
k1 = k1+Psi1/(2*pi);
%k2是delta2_Phi(i,j)/2pi
k2 = k2+Psi2/(2*pi);
%% 积分获取解缠相位
K = cumsum([0 k2(1,:)]);
K = [K; k1];
K = cumsum(K,1);
%解缠相位,K为解缠相位(真实相位)偏导(梯度)
PhiUnwrap = (2*pi)*K ;
end % 解缠处理主函数
%% 最小费用网络流子块解缠函数
function [x1p x1m x2p x2m] = cunwrap_blk(Psi, ...
                                         X1p, X1m, X2p, X2m, options)
% if getoption(options, 'blknum', NaN) == 8
%     save('debug.mat', 'Psi', 'X1p', 'X1m', 'X2p', 'X2m', 'options');
% end
%% 计算偏导数（行方向y、列方向x）
% ny:行,nx:列
[ny nx] = size(Psi);%缠绕相位尺寸
% the width (in pixel) of the Gaussian kernel to limit effect of
% patch that does not satisfy rotational relation
CutSize = getoption(options, 'cutsize', 4);
% Default weight
w1 = ones(ny,1); w1([1 end])=0.5;
w2 = ones(1,nx); w2([1 end])=0.5;
weight = w1*w2; % tensorial product
weight = getoption(options, 'weight', weight);
% Compute partial derivative Psi1, eqt (1,3)
i = 1:(ny-1);
j = 1:nx;
[ROW_I ROW_J] = ndgrid(i,j);%ndgrid:Generate arrays for N-D functions and interpolation
I_J = sub2ind(size(Psi),ROW_I,ROW_J);% sub2ind:线性化阵列下标
IP1_J = sub2ind(size(Psi),ROW_I+1,ROW_J);%
Psi1 = Psi(IP1_J) - Psi(I_J);%第2行-第1行;第3行-第2行;……;第ny行-第ny-1行;
Psi1 = mod(Psi1+pi,2*pi)-pi;
% Compute partial derivative Psi2, eqt (2,4)
i = 1:ny;
j = 1:(nx-1);
[ROW_I ROW_J] = ndgrid(i,j);
I_J = sub2ind(size(Psi),ROW_I,ROW_J);
I_JP1 = sub2ind(size(Psi),ROW_I,ROW_J+1);
Psi2 = Psi(I_JP1) - Psi(I_J);
Psi2 = mod(Psi2+pi,2*pi)-pi;
%% 约束条件（等式右侧）
% The RHS is column-reshaping of a 2D array [ny-1] x [nx-1]
% 计算式(17)所示约束条件的等式右侧（right-hand）
beq = 0;
% Compute beq
i = 1:(ny-1);
j = 1:(nx-1);
[ROW_I ROW_J] = ndgrid(i,j);
I_J = sub2ind(size(Psi1),ROW_I,ROW_J);
I_JP1 = sub2ind(size(Psi1),ROW_I,ROW_J+1);
beq = beq + (Psi1(I_JP1)-Psi1(I_J));
I_J = sub2ind(size(Psi2),ROW_I,ROW_J);
IP1_J = sub2ind(size(Psi2),ROW_I+1,ROW_J);
beq = beq - (Psi2(IP1_J)-Psi2(I_J));
beq = -1/(2*pi)*beq;
beq = round(beq);%round向最近整数取整
beq = beq(:);    %2维矩阵转换为列矩阵，多列并一列
%% 确定线性规划输入（LP：linear Programing)
% The vector of LP is arranged as following: 
% x := (x1p, x1m, x2p, x2m).'
% x1p, x1m: reshaping of [ny-1] x [nx]
% x2p, x2m: reshaping of [ny] x [nx-1]
% Row index, used by all foure blocks in Aeq, beq
i = 1:(ny-1);
j = 1:(nx-1);
[ROW_I ROW_J] = ndgrid(i,j);
ROW_I_J = sub2ind([length(i) length(j)],ROW_I,ROW_J);
nS0 = (nx-1)*(ny-1);%(i,j)的值域S0
% Use by S1p, S1m
[COL_I COL_J] = ndgrid(i,j);
COL_IJ_1 = sub2ind([length(i) length(j)+1],COL_I,COL_J);
[COL_I COL_JP1] = ndgrid(i,j+1);
COL_I_JP1 = sub2ind([length(i) length(j)+1],COL_I,COL_JP1);
nS1 = (nx)*(ny-1);%(i,j)的值域S1
% Use by S2p, S2m
[COL_I COL_J] = ndgrid(i,j);
COL_IJ_2 = sub2ind([length(i)+1 length(j)],COL_I,COL_J);
[COL_IP1 COL_J] = ndgrid(i+1,j);
COL_IP1_J = sub2ind([length(i)+1 length(j)],COL_IP1,COL_J);
nS2 = (nx-1)*(ny);%(i,j)的值域S2
% Build four indexes arrays that will be used to split x in four parts
offset1p = 0;
idx1p = offset1p+(1:nS1);
offset1m = idx1p(end);
idx1m = offset1m+(1:nS1);
offset2p = idx1m(end);
idx2p = offset2p+(1:nS2);
offset2m = idx2p(end);
idx2m = offset2m+(1:nS2);
% Equality constraint matrix (Aeq)
S1p = + sparse(ROW_I_J, COL_I_JP1,1,nS0,nS1) ...
      - sparse(ROW_I_J, COL_IJ_1,1,nS0,nS1);
S1m = -S1p; 
S2p = - sparse(ROW_I_J, COL_IP1_J,1,nS0,nS2) ...
      + sparse(ROW_I_J, COL_IJ_2,1,nS0,nS2);  
S2m = -S2p;
%% 约束条件（等式左侧）
Aeq = [S1p S1m S2p S2m];
nvars = size(Aeq,2);
% Clean up
clear S1p S1m S2p S2m
%% 权重
% 强制偶数转换
CutSize = ceil(CutSize/2)*2; 
% Modify weight to limit the effect of points that violate
% the rorational condition. The weight is graduataly increase
% around the points.
if CutSize>0
    % 截断高斯内核
    v = 1*linspace(-1,1,CutSize);
    [x y] = meshgrid(v,v);
    kernel = 1.1*exp(-(x.^2+y.^2));
    rotdegradation = double(reshape(beq~=0, [ny nx]-1)); % 0 or 1
    rotdegradation = conv2(rotdegradation, kernel, 'full');
    rotdegradation = rotdegradation(CutSize/2 + (0:ny-1),...
                                    CutSize/2 + (0:nx-1));
    % 权重下限     
    wmin = 1e-2; % weight >= win
    rotdegradation = min(rotdegradation,1-wmin);
    weight = weight .* (1-rotdegradation);
end
c1 = 0.5*(weight(1:ny-1,:)+weight(1:ny-1,:));
c2 = 0.5*(weight(:,1:nx-1)+weight(:,1:nx-1));
%% 费用流, eqt (16)
cost = zeros(nvars,1);
cost(idx1p) = c1(:);
cost(idx1m) = c1(:);
cost(idx2p) = c2(:);
cost(idx2m) = c2(:);
%% 上下限确定， eqt (18,19)
L = zeros(nvars,1);% Lower bound
U = Inf(size(L));  % No upper bound, U=[];
%% 锁定x值
% Lock the x values to prior computed value (from calculation on other
% blocks)
i1 = find(~isnan(X1p));
i2 = find(~isnan(X2p));     
% Lock method, not documented
lockadd = 1;
lockremove = 2; %#ok
lockmethod = getoption(options, 'lockmethod', lockadd);
if lockmethod==lockadd
    % Lock matrix and values, eqt (26, 27), larger system
    L1p = sparse(1:length(i1), idx1p(i1), 1, length(i1), size(Aeq,2));
    L1m = sparse(1:length(i1), idx1m(i1), 1, length(i1), size(Aeq,2));
    L2p = sparse(1:length(i2), idx2p(i2), 1, length(i2), size(Aeq,2));
    L2m = sparse(1:length(i2), idx2m(i2), 1, length(i2), size(Aeq,2));    
    AL = [L1p; L1m; L2p; L2m];
    bL = [X1p(i1); X1m(i1); X2p(i2); X2m(i2)];
    clear L1p L1m L2p L2m % clean up   
    % Find the rows in Aeq with all variables locked
    ColPatch = [offset1p+COL_IJ_1(:) offset1p+COL_I_JP1(:) ...
                offset2p+COL_IJ_2(:) offset2p+COL_IP1_J(:)];    
    [trash ColDone] = find(AL); %#ok
    remove = all(ismember(ColPatch, ColDone),2);
    Aeq = [Aeq(~remove,:); AL];
    beq = [beq(~remove,:); bL];
    % No need to bother with what already computed
    cost(idx1p(i1)) = 0;
    cost(idx1m(i1)) = 0;
    cost(idx2p(i2)) = 0;
    cost(idx2m(i2)) = 0;   
    L(idx1p(i1)) = -Inf;
    L(idx1m(i1)) = -Inf;
    L(idx2p(i2)) = -Inf;
    L(idx2m(i2)) = -Inf;
    clear AL bL trash ColPatch ColDone remove i1 i2 
else
    % Lock by remove the overlapped variables, smaller system
    % But *seems* more affected by error propagation
    % BL think both method should be strictly equivalent (!)   
    % remove the equality contribution from the RHS
    lock = zeros(nvars,1,class(Aeq));
    lock(idx1p(i1)) = X1p(i1);
    lock(idx1m(i1)) = X1m(i1);
    lock(idx2p(i2)) = X2p(i2);
    lock(idx2m(i2)) = X2m(i2);
    beq = beq - Aeq*lock;   
    % Remove the variables
    vremove = [idx1p(i1) idx1m(i1) idx2p(i2) idx2m(i2)];
    % keep is use later to dispatch partial derivative
    keep = true(nvars,1); keep(vremove) = false;
    Aeq(:,vremove) = [];
    L(vremove) = []; U(vremove) = [];
    cost(vremove) = [];  
    % Find the rows in Aeq with all variables locked
    ColPatch = [offset1p+COL_IJ_1(:) offset1p+COL_I_JP1(:) ...
                offset2p+COL_IJ_2(:) offset2p+COL_IP1_J(:)];
    % Remove the equations
    eremove = all(ismember(ColPatch, vremove),2);
    Aeq(eremove,:) = [];
    beq(eremove,:) = [];   
    clear vremove eremove ColPatch lock % clean up
end
%% LP solver
% To do: implement Bertsekas/Tseng's relaxation method, ref. [20]
% Call LP solver
if ~isempty(which('linprog'))
    % Call Matlab Linprog
    % http://www.mathworks.com/access/helpdesk/help/toolbox/optim/ug/linprog.html
    % http://www.mathworks.com/access/helpdesk/help/toolbox/optim/ug/optimset.html    
    % BL has not checked because he does not have the right toolbox
    LPoption = getoption(options, 'LPoption', {});
    if ~iscell(LPoption)
        LPoption = {LPoption};
    end
    % LPoption = {optimset(...)}
    sol = linprog(cost,[],[],Aeq,beq,L,U,[],LPoption{:});
elseif ~isempty(which('BuildMPS'))
    % Here is BL Linprog, call Matlab linprog instead to get "SOL",
    % the solution of the above LP problem
    mpsfile='costantini.mps';
    Contain = BuildMPS([], [], Aeq, beq, cost, L, U, 'costantini');
    OK = SaveMPS(mpsfile,Contain);
    if ~OK
        error('CUNWRAP: Cannot write mps file');
    end
    PCxpath=App_path('PCx');
    [OK outfile]=invokePCx(mpsfile,PCxpath,verbose==0);
    if ~OK
        mydisplay(verbose, 'PCx path=%s\n', PCxpath);
        mydisplay(verbose, 'Cannot invoke LP solver, PCx not installed?\n');
        error('CUNWRAP: Cannot invoke PCx');
    end
    [OK sol]=readPCxoutput(outfile);
    if ~OK
        error('CUNWRAP: Cannot read PCx outfile, L1 fit might fails.');
    end
else
    error('CUNWRAP: cannot detect network flow (LP) engine');
end
%% 分配LP结果
if lockmethod==lockadd
    x = sol;
else
    x = zeros(size(keep),class(sol));
    x(keep) = sol;
end
x1p = reshape(x(idx1p), [ny-1 nx]);
x1m = reshape(x(idx1m), [ny-1 nx]);
x2p = reshape(x(idx2p), [ny nx-1]);
x2m = reshape(x(idx2m), [ny nx-1]);
end
%% 读取option函数
function value = getoption(options, name, defaultvalue)
% function value = getoption(options, name, defaultvalue)
    %Input:options,name,defaultvalue(默认输入)
    %Output:value
    value = defaultvalue;%当无输入时，将默认输入赋值输出
    fields = fieldnames(options);
    found = strcmpi(name,fields);%strcmpi:比较字符串，当字符串一致，返回1
    if any(found)%any:判断输入是否非零，如果非零则返回1，反之返回0
        i = find(found,1,'first');%参数found是一组数组，find:返回found中第一个值为1的下标（位置）
        if ~isempty(options.(fields{i}))%当有输入时，将输入赋值输出
            value = options.(fields{i});
        end
    end
end
%% 分块函数
function [ilist blocksize] = splitidx(blocksize, n, p)
% function ilist = splitidx(blocksize, n, p)
% return the cell array, each element is subindex of (1:n)
% The union is (1:n) with overlapping fraction is p (0<p<1)
if blocksize>=n
    ilist = {1:n};%子块中元素下标
    blocksize = n;%子块一维尺寸
else
    q = 1-p;
    % Number of blocks,子块个数
    k = (n/blocksize - p) / q;
    k = ceil(k);%ceil向右（正无穷）取整
    % Readjust the block size, float，更新子块一维尺寸
    blocksize = n/(k*q + p);
    % first index各块起点下标
    firstidx = round(linspace(1,n-ceil(blocksize)+1, k));%round,向最近整数取整
    lastidx = round(firstidx+blocksize-1);
    lastidx(end) = n;
    % Make sure they are overlapped，确保块与块之间重叠
    % 各块终点下标
    lastidx(1:end-1) = max(lastidx(1:end-1),firstidx(2:end));
    % Put the indexes of k blocks into cell array
    % 输出各子块下标cell（单元）
    ilist = cell(1,length(firstidx));
    for k=1:length(ilist)
        ilist{k} = firstidx(k):lastidx(k);
    end
    % 输出子块一维尺寸
    blocksize = round(blocksize);
end
end