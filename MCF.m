function PhiUnwrap = MCF(Psi, options)
% �㷨: Minimum Network Flow
%
% INPUTS:
%   - Psi: 2ά������λ(rad)
% OUTPUT:
%   - Phi: 2ά�����λ(rad)
%% �������
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
%�в�ֵ����
roundK = getoption(options, 'roundk', false);
% Ĭ��Ȩ��(Default weight)
w1 = ones(ny,1); w1([1 end]) = 0.5;
w2 = ones(1,nx); w2([1 end]) = 0.5;
weight = w1*w2; % tensorial product������
weight = getoption(options, 'weight', weight);
% �ֿ�
% ���ֿ����Ϻ�(?)-------------20150415
bs = max(ny, nx);
blocksize = getoption(options, 'maxblocksize', bs);%maxblocksize���ӿ��С
blocksize = max(blocksize,2);
% �����֮���ص�����(25%)
p = getoption(options, 'overlap', 0.25);
p = max(min(p,1),0);
% �ӿ�����
[ix blk_x] = splitidx(blocksize, nx, p);%blk_x ���з����ӿ�ߴ磨���ظ�������ix���з����ӿ��±����м�
[iy blk_y] = splitidx(blocksize, ny, p);%blk_y ���з����ӿ�ߴ磨���ظ�������iy���з����ӿ��±����м�
%�ӿ����
nbblk = length(iy)*length(ix);
% %% write parameters
% fid1 = fopen('.\Mid\MCF_unwrap_parameters.txt','wt');
% fprintf(fid1,'==========���Ƹ�����λ��Ϣ=========\n');
% fprintf(fid1,'%s %i \n','������λ�ߴ�(��): ',ny);
% fprintf(fid1,'%s %i \n','������λ�ߴ�(��): ',nx);
% fprintf(fid1,'==========MCF���������Ϣ==========\n');
% fprintf(fid1,'%s %i \n','�ӿ����:     ',nbblk);
% fprintf(fid1,'%s %i \n','�ӿ�ߴ�(��): ',blk_y);
% fprintf(fid1,'%s %i \n','�ӿ�ߴ�(��): ',blk_x);
% fprintf(fid1,'%s %i %s \n','�ӿ��ص���:   ',p*100,'%');
% fclose(fid1);
%% �������С�������λƫ�������ܸ�����
% negative/positive parts of wrapped partial derivatives��
x1p = nan(ny-1, nx, class(Psi));%nan,;class:���ز����������������
x1m = nan(ny-1, nx, class(Psi));
x2p = nan(ny, nx-1, class(Psi));
x2m = nan(ny, nx-1, class(Psi));
%% �ӿ������Ƶ�ѭ�����ֿ���
blknum = 0;
for i=1:length(iy)%length(iy)���ӿ����
    iy0 = iy{i};%�����ӿ�����յ㼯
    iy1 = iy0(1:end-1);
    iy2 = iy0(1:end);    
    for j=1:length(ix)%length(ix)���ӿ����
        ix0 = ix{j};%�����ӿ�����յ㼯
        ix1 = ix0(1:end);
        ix2 = ix0(1:end-1);       
        blknum = blknum + 1;
        options.weight = weight(iy0,ix0);%����Ȩ�ؾ���
        options.blknum = blknum;         %�����ӿ����
        % Psi(iy0,ix0)�Ƿֿ��Ӧ�Ĳ�����λ,RELAX�������������С������
        % cunwrap_blk������С������x1p,x1m,x2p,x2m
        % ͨ��x1p,x1m,x2p,x2m����k1,k2���Ʋв�
        [x1p(iy1,ix1) x1m(iy1,ix1) x2p(iy2,ix2) x2m(iy2,ix2)] = ...
            cunwrap_blk(Psi(iy0,ix0), ...
            x1p(iy1,ix1), x1m(iy1,ix1),...
            x2p(iy2,ix2), x2m(iy2,ix2),...
            options);
    end
end
%% �����з���y-direction��ƫ����
% Compute partial derivative Psi1, eqt (1,3)
i = 1:(ny-1);
j = 1:nx;
[ROW_I ROW_J] = ndgrid(i,j);
I_J = sub2ind(size(Psi),ROW_I,ROW_J);
IP1_J = sub2ind(size(Psi),ROW_I+1,ROW_J);
Psi1 = Psi(IP1_J) - Psi(I_J);
Psi1 = mod(Psi1+pi,2*pi)-pi;%Psi1 is in [-pi,pi)
%% �����з���x-direction��ƫ����
% Compute partial derivative Psi2, eqt (2,4)
i = 1:ny;
j = 1:(nx-1);
[ROW_I ROW_J] = ndgrid(i,j);
I_J = sub2ind(size(Psi),ROW_I,ROW_J);
I_JP1 = sub2ind(size(Psi),ROW_I,ROW_J+1);
Psi2 = Psi(I_JP1) - Psi(I_J);
Psi2 = mod(Psi2+pi,2*pi)-pi;%Psi2 is in [-pi,pi)
%% ����ƫ���������, eqt (20,21)
% ͨ��x1p,x1m,x2p,x2m������Сƫ���в�k1,k2
k1 = x1p-x1m;
k2 = x2p-x2m;
% Round to integer solution (?)
% ������k1,k2
if roundK
    k1 = round(k1);
    k2 = round(k2);
end
%% ƫ������������
% Sum the jumps with the wrapped partial derivatives, eqt (10,11)
% ��������λ(��ʵ��λ)ƫ��(�ݶ�),��delta1_Phi(i,j)��delta2_Phi(i,j)
k1 = reshape(k1,[ny-1 nx]);
k2 = reshape(k2,[ny nx-1]);
%k1��delta1_Phi(i,j)/2pi
k1 = k1+Psi1/(2*pi);
%k2��delta2_Phi(i,j)/2pi
k2 = k2+Psi2/(2*pi);
%% ���ֻ�ȡ�����λ
K = cumsum([0 k2(1,:)]);
K = [K; k1];
K = cumsum(K,1);
%�����λ,KΪ�����λ(��ʵ��λ)ƫ��(�ݶ�)
PhiUnwrap = (2*pi)*K ;
end % �������������
%% ��С�����������ӿ�������
function [x1p x1m x2p x2m] = cunwrap_blk(Psi, ...
                                         X1p, X1m, X2p, X2m, options)
% if getoption(options, 'blknum', NaN) == 8
%     save('debug.mat', 'Psi', 'X1p', 'X1m', 'X2p', 'X2m', 'options');
% end
%% ����ƫ�������з���y���з���x��
% ny:��,nx:��
[ny nx] = size(Psi);%������λ�ߴ�
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
I_J = sub2ind(size(Psi),ROW_I,ROW_J);% sub2ind:���Ի������±�
IP1_J = sub2ind(size(Psi),ROW_I+1,ROW_J);%
Psi1 = Psi(IP1_J) - Psi(I_J);%��2��-��1��;��3��-��2��;����;��ny��-��ny-1��;
Psi1 = mod(Psi1+pi,2*pi)-pi;
% Compute partial derivative Psi2, eqt (2,4)
i = 1:ny;
j = 1:(nx-1);
[ROW_I ROW_J] = ndgrid(i,j);
I_J = sub2ind(size(Psi),ROW_I,ROW_J);
I_JP1 = sub2ind(size(Psi),ROW_I,ROW_J+1);
Psi2 = Psi(I_JP1) - Psi(I_J);
Psi2 = mod(Psi2+pi,2*pi)-pi;
%% Լ����������ʽ�Ҳࣩ
% The RHS is column-reshaping of a 2D array [ny-1] x [nx-1]
% ����ʽ(17)��ʾԼ�������ĵ�ʽ�Ҳࣨright-hand��
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
beq = round(beq);%round���������ȡ��
beq = beq(:);    %2ά����ת��Ϊ�о��󣬶��в�һ��
%% ȷ�����Թ滮���루LP��linear Programing)
% The vector of LP is arranged as following: 
% x := (x1p, x1m, x2p, x2m).'
% x1p, x1m: reshaping of [ny-1] x [nx]
% x2p, x2m: reshaping of [ny] x [nx-1]
% Row index, used by all foure blocks in Aeq, beq
i = 1:(ny-1);
j = 1:(nx-1);
[ROW_I ROW_J] = ndgrid(i,j);
ROW_I_J = sub2ind([length(i) length(j)],ROW_I,ROW_J);
nS0 = (nx-1)*(ny-1);%(i,j)��ֵ��S0
% Use by S1p, S1m
[COL_I COL_J] = ndgrid(i,j);
COL_IJ_1 = sub2ind([length(i) length(j)+1],COL_I,COL_J);
[COL_I COL_JP1] = ndgrid(i,j+1);
COL_I_JP1 = sub2ind([length(i) length(j)+1],COL_I,COL_JP1);
nS1 = (nx)*(ny-1);%(i,j)��ֵ��S1
% Use by S2p, S2m
[COL_I COL_J] = ndgrid(i,j);
COL_IJ_2 = sub2ind([length(i)+1 length(j)],COL_I,COL_J);
[COL_IP1 COL_J] = ndgrid(i+1,j);
COL_IP1_J = sub2ind([length(i)+1 length(j)],COL_IP1,COL_J);
nS2 = (nx-1)*(ny);%(i,j)��ֵ��S2
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
%% Լ����������ʽ��ࣩ
Aeq = [S1p S1m S2p S2m];
nvars = size(Aeq,2);
% Clean up
clear S1p S1m S2p S2m
%% Ȩ��
% ǿ��ż��ת��
CutSize = ceil(CutSize/2)*2; 
% Modify weight to limit the effect of points that violate
% the rorational condition. The weight is graduataly increase
% around the points.
if CutSize>0
    % �ضϸ�˹�ں�
    v = 1*linspace(-1,1,CutSize);
    [x y] = meshgrid(v,v);
    kernel = 1.1*exp(-(x.^2+y.^2));
    rotdegradation = double(reshape(beq~=0, [ny nx]-1)); % 0 or 1
    rotdegradation = conv2(rotdegradation, kernel, 'full');
    rotdegradation = rotdegradation(CutSize/2 + (0:ny-1),...
                                    CutSize/2 + (0:nx-1));
    % Ȩ������     
    wmin = 1e-2; % weight >= win
    rotdegradation = min(rotdegradation,1-wmin);
    weight = weight .* (1-rotdegradation);
end
c1 = 0.5*(weight(1:ny-1,:)+weight(1:ny-1,:));
c2 = 0.5*(weight(:,1:nx-1)+weight(:,1:nx-1));
%% ������, eqt (16)
cost = zeros(nvars,1);
cost(idx1p) = c1(:);
cost(idx1m) = c1(:);
cost(idx2p) = c2(:);
cost(idx2m) = c2(:);
%% ������ȷ���� eqt (18,19)
L = zeros(nvars,1);% Lower bound
U = Inf(size(L));  % No upper bound, U=[];
%% ����xֵ
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
%% ����LP���
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
%% ��ȡoption����
function value = getoption(options, name, defaultvalue)
% function value = getoption(options, name, defaultvalue)
    %Input:options,name,defaultvalue(Ĭ������)
    %Output:value
    value = defaultvalue;%��������ʱ����Ĭ�����븳ֵ���
    fields = fieldnames(options);
    found = strcmpi(name,fields);%strcmpi:�Ƚ��ַ��������ַ���һ�£�����1
    if any(found)%any:�ж������Ƿ���㣬��������򷵻�1����֮����0
        i = find(found,1,'first');%����found��һ�����飬find:����found�е�һ��ֵΪ1���±꣨λ�ã�
        if ~isempty(options.(fields{i}))%��������ʱ�������븳ֵ���
            value = options.(fields{i});
        end
    end
end
%% �ֿ麯��
function [ilist blocksize] = splitidx(blocksize, n, p)
% function ilist = splitidx(blocksize, n, p)
% return the cell array, each element is subindex of (1:n)
% The union is (1:n) with overlapping fraction is p (0<p<1)
if blocksize>=n
    ilist = {1:n};%�ӿ���Ԫ���±�
    blocksize = n;%�ӿ�һά�ߴ�
else
    q = 1-p;
    % Number of blocks,�ӿ����
    k = (n/blocksize - p) / q;
    k = ceil(k);%ceil���ң������ȡ��
    % Readjust the block size, float�������ӿ�һά�ߴ�
    blocksize = n/(k*q + p);
    % first index��������±�
    firstidx = round(linspace(1,n-ceil(blocksize)+1, k));%round,���������ȡ��
    lastidx = round(firstidx+blocksize-1);
    lastidx(end) = n;
    % Make sure they are overlapped��ȷ�������֮���ص�
    % �����յ��±�
    lastidx(1:end-1) = max(lastidx(1:end-1),firstidx(2:end));
    % Put the indexes of k blocks into cell array
    % ������ӿ��±�cell����Ԫ��
    ilist = cell(1,length(firstidx));
    for k=1:length(ilist)
        ilist{k} = firstidx(k):lastidx(k);
    end
    % ����ӿ�һά�ߴ�
    blocksize = round(blocksize);
end
end