%   This code comes with the research paper entitled 
%
%   "Structure-informed functional connectivity driven by identifiable and state-specific control regions"
%       by Benjamin Chiem, Frederic Crevecoeur and Jean-Charles Delvenne (2020)
%
%   INPUTS :
%               - S      : Adjacency matrix of the connectome
%               - F_emp  : empirical functional connectivity matrix
%               - params : structure of parameters
%                   - params.dynamics : describes the communication
%                       dynamics ('DIFF' or 'ADJ')
%                   - params.nTests : number of optimization runs
%                   - params.constrained : boolean indicating whether the
%                       optimization problem is constrained or not
%                   - params.useParallel : boolean indicating whether or
%                       not to use parallel computing
%                   - params.verbose : boolean indicating whether or not to
%                       display intermediate results
%
%   OUTPUTS :
%               - corr_score(i,j) contains the correlation score for test
%                   i and constraint j
%               - control_ROIs(i,:,j) is a binary indicator vector encoding 
%                   which ROIs have been selected for test i and constraint j
%
%   Please note that this function requires the Matlab implementation of
%   genetic algorithm (function ga()), from the Global Optimisation Toolbox
%   
%   Author : Benjamin Chiem (benjamin.chiem@uclouvain.be)
%            Universite catholique de Louvain

function [corr_score,control_ROIs] = identify_control_ROIs(S,F_emp,params)

% Sanity checks
assert(size(S,1)==size(S,2)&&size(F_emp,1)==size(F_emp,2),'Input matrices not square');
assert(isequal(size(S),size(F_emp)),'Input matrices do not have matching dimensions');

%% Parse parameters

nROIs = size(S,1);

% Sets the default dynamics to DIFF (Laplacian diffusion dynamics)
if ~isfield(params,'dynamics')
    params.dynamics = 'DIFF';
end

% Sets the default number of runs of the heuristic optimisation 
if ~isfield(params,'nTests')
    params.nTests = 30;
end

% Sets the default usage of cardinality constraints
if ~isfield(params,'constrained')
    params.constrained = false;
end

% Sets the default usage of parallel computing
if ~isfield(params,'useParallel')
    params.useParallel = true;
end

% Sets default value for verbose
if ~isfield(params,'verbose')
    params.verbose = true;
end

%% Model settings

S = S-diag(diag(S)); % Remove diagonal entries in S

if strcmp(params.dynamics,'DIFF')
    D    = diag(sum(S,2));           % Degree matrix
    L    = D^(-0.5)*(D-S)*D^(-0.5);  % Normalized Laplacian
    beta = 0.72;                     % Discretization parameter
    A    = expm(-beta*L);            % Communication matrix (see Abdelnour et al, 2014)
elseif strcmp(params.dynamics,'ADJ')
    A    = S;                        % Communication matrix (see Gu et al, 2015)
else
    error('Unknown dynamics. Choices are : DIFF or ADJ');
end

A = A./(1+svds(A,1)); % Stabilized system matrix (see Karrer et al, 2020)

% Define the fitness function for the genetic algorithm (heuristic optimisation)
% (-1)* is to obtain a MAX problem
FitnessFunction = @(inputs) ((-1)*compute_correlation_score(inputs,A,F_emp));

if params.constrained
    U        = 1:nROIs;          % Upper bound on the number of selected ROIs
else
    U        = nROIs;
end

nConstraints = length(U);
opt_vec      = ones(1,nROIs);    % Utility vector to define the optimisation constraint

% The three following lines will enforce the optimisation variables to be binary
lb           = zeros(nROIs,1);
ub           = ones(nROIs,1);
intCon       = 1:nROIs;

corr_score   = zeros(params.nTests,nConstraints);
control_ROIs = zeros(params.nTests,nROIs,nConstraints);

%% Optimization
options = optimoptions('ga','Display','off','PopulationSize',200,...
    'MaxStallGenerations',50,'UseParallel',params.useParallel,'UseVectorized',false);

for iCon = 1:nConstraints
    for iTest = 1:params.nTests
        rng(iTest*iCon); % Sets seed for reproducibility
        if params.verbose
            fprintf('>>>>> Test %i over %i with %i input ROIs allowed ... \n',iTest,params.nTests,U(iCon));
        end
        [control_ROIs(iTest,:,iCon),corr_score(iTest,iCon)] = ga(FitnessFunction,...
            nROIs,opt_vec,U(iCon),[],[],lb,ub,[],intCon,options);
        if params.verbose
            fprintf('Correlation score : %.4f \n',-corr_score(iTest,iCon));
        end
    end
end


end


% Objective function for the heuristic optimization
% Here : statistical similarity, i.e. Pearson's correlation between
% vectorized upper-triangular parts of both matrices.
function res = compute_correlation_score(inputs,A,F_emp)
    
    nROIs    = size(F_emp,1);
    mask_ut  = triu(true(nROIs,nROIs),1);    % Upper-triangular mask
    canonVec = eye(nROIs,nROIs);             % Canonical vectors
    B        = canonVec(:,logical(inputs));  % Input matrix
    Sig      = dlyap(A,B*B');                % Controllability Gramian
    P        = diag(diag(Sig));              % Diagonal matrix containing state variances
    F_si     = P^(-0.5)*Sig*P^(-0.5);        % Structure-informed functional connectivity
    
    res      = corr(F_si(mask_ut),F_emp(mask_ut));
end