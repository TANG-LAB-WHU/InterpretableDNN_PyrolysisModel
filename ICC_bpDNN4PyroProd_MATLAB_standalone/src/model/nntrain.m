% function [net, tr] = nntrain(net, P, T)
%     %NNTRAIN trains a network with P inputs and T targets according to the
%     %       training strategy which are defined in net.
%     %   Inputs:
%     %           net - a feedforward neural network created by NNCREATE.
%     %           P - Inputs of networks. Inputs could be the measurement of
%     %           samples. Each column is a sample and each row is a feature. The
%     %           number of colomns are the number of samples.
%     %           T - Targets of outputs. Each column is the target of a samples
%     %           and each row is an output. The number of colomons is equal to
%     %           the number of samples. The number of rows is equal to outputs.
%     %   Outputs:
%     %           net - A trained network.
%     %           tr - Training record containing performance information.
%     %   Example:
%     %           [net, tr] = NNTRAIN(net, P, T)
    
%     % Data configuring.
%     [net, data] = nnconfigure(net, P, T);
    
%     % Data preparation.
%     % Ensure nnprepare correctly returns PS and TS if needed later
%     % Assuming PS and TS might be fields within 'net' or 'data' after this call
%     % [net, tr, data, option, PS, TS] = nnprepare(net, data); % Assuming nnprepare can return PS, TS
%     [net, tr, data, option] = nnprepare(net, data); 
%     % Make sure tr is initialized
%     if ~exist('tr', 'var') || isempty(tr)
%         tr = struct();
%         tr.trainInd = [];
%         tr.valInd = [];
%         tr.testInd = [];
%         tr.perf = [];
%         tr.vperf = [];
%         tr.tperf = [];
%         tr.best_perf = [];
%         tr.best_vperf = [];
%         tr.best_tperf = [];
%         tr.epoch = [];
%         tr.time = [];
%         tr.gradient = [];  
%         tr.stop = [];
%         tr.best_epoch = [];
%         tr.num_epochs = [];
%         tr.val_fail = [];
%     end
    
%     t = cputime;
    
%     % Initialize error.
%     error = [];
    
%     N = size(data.P, 2);
    
%     % Assign input and outputs.
%     trainInd = data.trainInd;
%     valInd = data.valInd;
%     testInd = data.testInd;
    
%     Ptrain = data.P(:, trainInd);
%     Ttrain = data.T(:, trainInd);
    
%     validation = option.validation;
%     testing = option.testing;
%     Pval = []; Tval = [];
%     Ptest = []; Ttest = [];
    
%     if validation
%         Pval = data.P(:, valInd);
%         Tval = data.T(:, valInd);
%     end
    
%     if testing
%         Ptest = data.P(:, testInd);
%         Ttest = data.T(:, testInd);
%     end
    
%     epoch = 0;
%     val_fail = 0;
    
%     pdW = data.pdW;
%     pdb = data.pdb;
    
%     fig = data.figure;
%     fail = 0;
%     best_vperf = inf;
%     best_perf = inf;
%     best_tperf = inf;
    
%     best_net = net;
    
%     best_epoch = 0;
    
%     max_fail = net.trainParam.max_fail;
%     goal = net.trainParam.goal;
%     epochs = net.trainParam.epoch; % Corrected from 'epochs = net.trainParam.epoch;' to use correct field name if different
%     min_grad = net.trainParam.min_grad;
    
%     while 1
%         epoch = epoch + 1;
    
%         fprintf('DEBUG nntrain Loop (Epoch %d): Start\n', epoch); % Add epoch marker
    
%         % Call nnff, getting the cell array of all layer activations for training set
%         Atrain_all_layers = nnff(net, Ptrain); 
    
%         % --- Extract final output for training set ---
%         Otrain = Atrain_all_layers{net.numLayer}; 
%         % ---
    
%         fprintf('DEBUG nntrain Loop: Size Ptrain = [%d, %d]\n', size(Ptrain,1), size(Ptrain,2)); 
%         fprintf('DEBUG nntrain Loop: Size Otrain (extracted output) = [%d, %d]\n', size(Otrain,1), size(Otrain,2)); 
%         fprintf('DEBUG nntrain Loop: Size Ttrain = [%d, %d]\n', size(Ttrain,1), size(Ttrain,2)); 
    
%         % Check sizes BEFORE subtraction
%         if ~isequal(size(Ttrain), size(Otrain))
%             fprintf('ERROR nntrain Loop: Size mismatch DETECTED before Ttrain - Otrain! Ttrain:[%d,%d], Otrain:[%d,%d]\n', ...
%                 size(Ttrain,1), size(Ttrain,2), size(Otrain,1), size(Otrain,2));
%             error('Stopping due to Ttrain/Otrain size mismatch.'); % Stop execution if sizes don't match
%         else
%             fprintf('DEBUG nntrain Loop: Ttrain and Otrain sizes match. Proceeding with error calculation.\n'); 
%         end
    
%         error = Ttrain - Otrain;
%         fprintf('DEBUG nntrain Loop: Size error = [%d, %d]\n', size(error,1), size(error,2)); 
    
%         % Backpropagation (BP) to calibrate weights and biases of the network.
%         fprintf('DEBUG nntrain Loop: Calling nnbp...\n');
%         % Pass the full cell array of activations to nnbp
%         [net, gradient, dW, db] = nnbp(net, Ptrain, Ttrain, error, Atrain_all_layers, pdW, pdb);
%         fprintf('DEBUG nntrain Loop: nnbp call completed.\n');
            
    
%         Etrain = evaluate_network(net.performFcn, error);
%         tr.perf = [tr.perf, Etrain];
%         tr.gradient = [tr.gradient, gradient];
%         tr.trainInd = [tr.trainInd, {trainInd}];
%         tr.valInd = [tr.valInd, {valInd}];
%         tr.testInd = [tr.testInd, {testInd}];
%         tr.epoch = [tr.epoch, epoch];
%         tr.time = [tr.time, cputime - t];
        
%         if gradient <= min_grad
%             tr.stop = [tr.stop, {'Minimum gradient reached.'}];
%             break
%         end
        
%         % Validation performance.
%         if validation
%             % Call nnff with one output for validation set
%             Aval_all_layers = nnff(net, Pval);
%             Oval = Aval_all_layers{net.numLayer}; % Extract final output
%             Eval = evaluate_network(net.performFcn, Tval - Oval);
%             tr.vperf = [tr.vperf, Eval];
%         end
        
%         % Test performance.
%         if testing
%             % Call nnff with one output for test set
%             Atest_all_layers = nnff(net, Ptest); 
%             % --- START Fix: Extract Otest from the single output argument ---
%             Otest = Atest_all_layers{net.numLayer}; 
%             % --- END Fix ---
    
%             fprintf('DEBUG nntrain (Test Eval): Size Ttest = [%d, %d]\n', size(Ttest,1), size(Ttest,2)); 
%             fprintf('DEBUG nntrain (Test Eval): Size Otest = [%d, %d]\n', size(Otest,1), size(Otest,2)); 
            
%             % Ensure sizes match before subtraction
%             if isequal(size(Ttest), size(Otest))
%                 Etest = evaluate_network(net.performFcn, Ttest - Otest); 
%                 tr.tperf = [tr.tperf, Etest];
%             else
%                 % This case should ideally not happen now, but keep error handling
%                 fprintf('ERROR: Size mismatch between Ttest and Otest in nntrain (Epoch %d)!\n', epoch); 
%                 Etest = NaN; % Assign NaN or handle error appropriately
%                 tr.tperf = [tr.tperf, Etest];
%                 tr.stop = [tr.stop, {'Test set evaluation size mismatch.'}]; % Add stop reason
%                 % break; % Consider breaking if test evaluation is critical and fails
%             end
%         else
%             Etest = NaN; % Ensure Etest exists even if testing is false
%         end
    
%         % Update train and validation errors for plotting.
%         if net.trainParam.showWindow
%             h = figure(fig);
%             if ~ishandle(h)
%                 fprintf('User closed the GUI window. Training process is interrupted.\n\n');
%                 tr.stop = [tr.stop, {'User closed the GUI window.'}];
%                 break
%             end
%             set(0, 'CurrentFigure', fig);
%             subplot(3,1,1); cla; hold on; % Clear previous plot
%             plot(tr.epoch, tr.perf, 'r-', 'LineWidth', 1.5);
%             legendEntries = {'Train'};
%             if validation && ~isempty(tr.vperf)
%                 plot(tr.epoch, tr.vperf, 'b:', 'LineWidth', 1.5);
%                 legendEntries{end+1} = 'Validation';
%             end
%             if testing && ~isempty(tr.tperf) && ~all(isnan(tr.tperf)) % Check if tperf has non-NaN values
%                 plot(tr.epoch, tr.tperf, 'g-.', 'LineWidth', 1.5);
%                 legendEntries{end+1} = 'Test';
%             end
%             title('Training Performance'); xlabel('Epoch'); ylabel(upper(net.performFcn));
%             legend(legendEntries, 'Location', 'northeast');
%             hold off;
    
%             subplot(3,1,2); cla; % Clear previous plot
%             plot(tr.epoch, tr.gradient); title('Gradient Magnitude'); xlabel('Epoch'); ylabel('Gradient');
            
%             subplot(3,1,3); cla; % Clear previous plot
%             if validation && length(tr.epoch) > 1
%                 plot(tr.epoch(2:end), diff(tr.vperf)); title('\Delta Validation Error'); xlabel('Epoch'); ylabel('\Delta Error');
%             elseif length(tr.epoch) > 1
%                 plot(tr.epoch(2:end), diff(tr.perf)); title('\Delta Training Error'); xlabel('Epoch'); ylabel('\Delta Error');
%             else
%                  title('\Delta Error (Requires >1 epoch)'); xlabel('Epoch'); ylabel('\Delta Error');
%             end
%             drawnow;
%         end
        
%         % Re-assign pdW, pdb; (crucial)
%         pdW = dW;
%         pdb = db;
        
%         % Update best train performance.
%         if Etrain < best_perf
%             best_perf = Etrain;
%         end
%         % Store current best performance at each epoch (simplifies logic)
%         tr.best_perf = [tr.best_perf, best_perf]; 
        
%         % Update best validation performance.
%         if validation
%             if Eval < best_vperf
%                 best_vperf = Eval;
%                 best_net = net;
%                 best_epoch = epoch;
%                 val_fail = 0;
%             else
%                 val_fail = val_fail + 1;
%             end
%             % Store current best validation performance and best epoch
%             tr.best_vperf = [tr.best_vperf, best_vperf];
%             tr.best_epoch = [tr.best_epoch, best_epoch];
%         else
%             % If no validation, keep track of best training performance epoch
%              if Etrain <= best_perf % Use <= to capture the first time best is hit
%                  best_epoch = epoch; % Update best epoch based on training perf
%                  best_net = net;    % Update best network based on training perf
%              end
%              tr.best_epoch = [tr.best_epoch, best_epoch]; % Store current best epoch
%         end
        
%         % Update best test performance.
%         if testing && ~isnan(Etest) % Only update if Etest is valid
%             if Etest < best_tperf
%                 best_tperf = Etest;
%             end
%             tr.best_tperf = [tr.best_tperf, best_tperf];
%         elseif testing % If testing but Etest is NaN
%              tr.best_tperf = [tr.best_tperf, best_tperf]; % Keep the previous best
%         end
        
%         tr.val_fail = [tr.val_fail, val_fail];
        
%         % --- Stop Criteria Checks ---
%         stopReason = ''; % Initialize empty stop reason for this epoch
%         if validation && val_fail >= max_fail
%             stopReason = 'Maximum validation failures reached.';
%         elseif Etrain <= goal % Check training performance against goal
%             stopReason = 'Performance goal met.';
%         elseif epoch >= epochs
%             stopReason = 'Maximum epoch reached.';
%         elseif gradient <= min_grad % Check minimum gradient (already checked earlier, but good to have here too)
%             stopReason = 'Minimum gradient reached.';
%         elseif ~isempty(tr.stop) && ~strcmp(tr.stop{end}, 'User closed the GUI window.') % Check if user closed window
%             stopReason = tr.stop{end}; % Propagate existing stop reason
%         end
    
%         if ~isempty(stopReason)
%             tr.stop = [tr.stop, {stopReason}];
%             fprintf('\nINFO: Stopping training. Reason: %s\n', stopReason);
%             break; % Exit the while loop
%         end
    
%         % --- Command Line Output ---
%         fprintf('Epoch: %d/%d (%.1f%%), TrainMSE: %.6f', epoch, epochs, epoch/epochs*100, Etrain);
%         if validation
%             fprintf(', ValMSE: %.6f (Best: %.6f, Fail: %d)', Eval, best_vperf, val_fail);
%         end
%         if testing && ~isnan(Etest)
%             fprintf(', TestMSE: %.6f (Best: %.6f)', Etest, best_tperf);
%         elseif testing
%             fprintf(', TestMSE: NaN');
%         end
%         fprintf(', Grad: %.6f\n', gradient);
    
%     end % End while loop
    
%     % Final assignments after loop
%     tr.num_epochs = epoch; % Record the actual number of epochs run
%     net = best_net; % Return the best network found (based on validation or training perf)
    
%     % Ensure final fields have consistent lengths if training stopped early
%     final_len = length(tr.epoch);
%     tr.perf = tr.perf(1:final_len);
%     tr.gradient = tr.gradient(1:final_len);
%     if validation, tr.vperf = tr.vperf(1:final_len); end
%     if testing, tr.tperf = tr.tperf(1:final_len); end
%     tr.best_perf = tr.best_perf(1:final_len);
%     if validation, tr.best_vperf = tr.best_vperf(1:final_len); end
%     if testing, tr.best_tperf = tr.best_tperf(1:final_len); end
%     tr.best_epoch = tr.best_epoch(1:final_len);
%     tr.val_fail = tr.val_fail(1:final_len);
%     tr.trainInd = tr.trainInd(1:final_len);
%     tr.valInd = tr.valInd(1:final_len);
%     tr.testInd = tr.testInd(1:final_len);
%     tr.time = tr.time(1:final_len);
    
%     % Assign the final best values to the main struct fields for convenience
%     tr.best_perf = best_perf;
%     tr.best_vperf = best_vperf;
%     tr.best_tperf = best_tperf;
%     tr.best_epoch = best_epoch; % Epoch where best net was saved
    
    
% end % End nntrain function
    
% %% Helper function to evaluate performance
% function perf = evaluate_network(performFcn, error)
%     % Helper function to evaluate performance based on error
%     switch lower(performFcn)
%         case 'mse'
%             % Calculate MSE, handle potential empty error matrix (e.g., if no test samples)
%             if isempty(error)
%                 perf = NaN;
%             else
%                 perf = mean(mean(error.^2));
%             end
%         case 'sse'
%             if isempty(error)
%                 perf = NaN;
%             else
%                 perf = sum(sum(error.^2));
%             end
%         case 'mae'
%             if isempty(error)
%                 perf = NaN;
%             else
%                 perf = mean(mean(abs(error)));
%             end
%         case 'sae'
%             if isempty(error)
%                 perf = NaN;
%             else
%                 perf = sum(sum(abs(error)));
%             end
%         otherwise
%             warning('NN:Performance:unknown', 'Unknown performance function: %s. Returning NaN.', performFcn);
%             perf = NaN; % Return NaN for unknown function
%     end
% end

function [net, tr] = nntrain(net, P, T)
    %NNTRAIN trains a network with P inputs and T targets according to the
    %       training strategy which are defined in net.
    %   Inputs:
    %           net - a feedforward neural network created by NNCREATE.
    %           P - Inputs of networks. Inputs could be the measurement of
    %           samples. Each column is a sample and each row is a feature. The
    %           number of colomns are the number of samples.
    %           T - Targets of outputs. Each column is the target of a samples
    %           and each row is an output. The number of colomons is equal to
    %           the number of samples. The number of rows is equal to outputs.
    %   Outputs:
    %           net - A trained network.
    %           tr - Training record containing performance information.
    %   Example:
    %           [net, tr] = NNTRAIN(net, P, T)
    
    % Data configuring.
    [net, data] = nnconfigure(net, P, T);
    
    % Data preparation.
    % Ensure nnprepare correctly returns PS and TS if needed later
    % Assuming PS and TS might be fields within 'net' or 'data' after this call
    % [net, tr, data, option, PS, TS] = nnprepare(net, data); % ORIGINAL Line causing error
    [net, tr, data, option] = nnprepare(net, data);
    
    % Make sure tr is initialized
    if ~exist('tr', 'var') || isempty(tr)
        tr = struct();
        tr.trainInd = [];
        tr.valInd = [];
        tr.testInd = [];
        tr.perf = [];
        tr.vperf = [];
        tr.tperf = [];
        tr.best_perf = [];
        tr.best_vperf = [];
        tr.best_tperf = [];
        tr.epoch = [];
        tr.time = [];
        tr.gradient = [];  
        tr.stop = [];
        tr.best_epoch = [];
        tr.num_epochs = [];
        tr.val_fail = [];
    end
    
    t = cputime;
    
    % Initialize error.
    error = [];
    
    N = size(data.P, 2);
    
    % Assign input and outputs.
    trainInd = data.trainInd;
    valInd = data.valInd;
    testInd = data.testInd;
    
    Ptrain = data.P(:, trainInd);
    Ttrain = data.T(:, trainInd);
    
    validation = option.validation;
    testing = option.testing;
    Pval = []; Tval = [];
    Ptest = []; Ttest = [];
    
    if validation
        Pval = data.P(:, valInd);
        Tval = data.T(:, valInd);
    end
    
    if testing
        Ptest = data.P(:, testInd);
        Ttest = data.T(:, testInd);
    end
    
    epoch = 0;
    val_fail = 0;
    
    pdW = data.pdW;
    pdb = data.pdb;
    
    fig = data.figure;
    fail = 0;
    best_vperf = inf;
    best_perf = inf;
    best_tperf = inf;
    
    best_net = net;
    
    best_epoch = 0;
    
    max_fail = net.trainParam.max_fail;
    goal = net.trainParam.goal;
    % Assuming trainParam.epochs exists and is correctly named
    epochs = net.trainParam.epochs; 
    min_grad = net.trainParam.min_grad;
    
    while 1
        epoch = epoch + 1;
    
        fprintf('DEBUG nntrain Loop (Epoch %d): Start\n', epoch); % Add epoch marker
    
        % Call nnff, getting the cell array of all layer activations for training set
        Atrain_all_layers = nnff(net, Ptrain); 
    
        % --- Extract final output for training set ---
        Otrain = Atrain_all_layers{net.numLayer}; 
        % ---
    
        fprintf('DEBUG nntrain Loop: Size Ptrain = [%d, %d]\n', size(Ptrain,1), size(Ptrain,2)); 
        fprintf('DEBUG nntrain Loop: Size Otrain (extracted output) = [%d, %d]\n', size(Otrain,1), size(Otrain,2)); 
        fprintf('DEBUG nntrain Loop: Size Ttrain = [%d, %d]\n', size(Ttrain,1), size(Ttrain,2)); 
    
        % Check sizes BEFORE subtraction
        if ~isequal(size(Ttrain), size(Otrain))
            fprintf('ERROR nntrain Loop: Size mismatch DETECTED before Ttrain - Otrain! Ttrain:[%d,%d], Otrain:[%d,%d]\n', ...
                size(Ttrain,1), size(Ttrain,2), size(Otrain,1), size(Otrain,2));
            error('Stopping due to Ttrain/Otrain size mismatch.'); % Stop execution if sizes don't match
        else
            fprintf('DEBUG nntrain Loop: Ttrain and Otrain sizes match. Proceeding with error calculation.\n'); 
        end
    
        error = Ttrain - Otrain;
        fprintf('DEBUG nntrain Loop: Size error = [%d, %d]\n', size(error,1), size(error,2)); 
    
        % Backpropagation (BP) to calibrate weights and biases of the network.
        fprintf('DEBUG nntrain Loop: Calling nnbp...\n');
        % Pass the full cell array of activations to nnbp
        [net, gradient, dW, db] = nnbp(net, Ptrain, Ttrain, error, Atrain_all_layers, pdW, pdb);
        fprintf('DEBUG nntrain Loop: nnbp call completed.\n');
            
    
        Etrain = evaluate_network(net.performFcn, error);
        tr.perf = [tr.perf, Etrain];
        tr.gradient = [tr.gradient, gradient];
        tr.trainInd = [tr.trainInd, {trainInd}];
        tr.valInd = [tr.valInd, {valInd}];
        tr.testInd = [tr.testInd, {testInd}];
        tr.epoch = [tr.epoch, epoch];
        tr.time = [tr.time, cputime - t];
        
        if gradient <= min_grad
            tr.stop = [tr.stop, {'Minimum gradient reached.'}];
            break
        end
        
        % Validation performance.
        if validation
            % Call nnff with one output for validation set
            Aval_all_layers = nnff(net, Pval);
            Oval = Aval_all_layers{net.numLayer}; % Extract final output
            Eval = evaluate_network(net.performFcn, Tval - Oval);
            tr.vperf = [tr.vperf, Eval];
        end
        
        % Test performance.
        if testing
            % Call nnff with one output for test set
            Atest_all_layers = nnff(net, Ptest); 
            % --- START Fix: Extract Otest from the single output argument ---
            Otest = Atest_all_layers{net.numLayer}; 
            % --- END Fix ---
    
            fprintf('DEBUG nntrain (Test Eval): Size Ttest = [%d, %d]\n', size(Ttest,1), size(Ttest,2)); 
            fprintf('DEBUG nntrain (Test Eval): Size Otest = [%d, %d]\n', size(Otest,1), size(Otest,2)); 
            
            % Ensure sizes match before subtraction
            if isequal(size(Ttest), size(Otest))
                Etest = evaluate_network(net.performFcn, Ttest - Otest); 
                tr.tperf = [tr.tperf, Etest];
            else
                % This case should ideally not happen now, but keep error handling
                fprintf('ERROR: Size mismatch between Ttest and Otest in nntrain (Epoch %d)!\n', epoch); 
                Etest = NaN; % Assign NaN or handle error appropriately
                tr.tperf = [tr.tperf, Etest];
                tr.stop = [tr.stop, {'Test set evaluation size mismatch.'}]; % Add stop reason
                % break; % Consider breaking if test evaluation is critical and fails
            end
        else
            Etest = NaN; % Ensure Etest exists even if testing is false
        end
    
        % Update train and validation errors for plotting.
        if net.trainParam.showWindow
            h = figure(fig);
            if ~ishandle(h)
                fprintf('User closed the GUI window. Training process is interrupted.\n\n');
                tr.stop = [tr.stop, {'User closed the GUI window.'}];
                break
            end
            set(0, 'CurrentFigure', fig);
            subplot(3,1,1); cla; hold on; % Clear previous plot
            plot(tr.epoch, tr.perf, 'r-', 'LineWidth', 1.5);
            legendEntries = {'Train'};
            if validation && ~isempty(tr.vperf)
                plot(tr.epoch, tr.vperf, 'b:', 'LineWidth', 1.5);
                legendEntries{end+1} = 'Validation';
            end
            if testing && ~isempty(tr.tperf) && ~all(isnan(tr.tperf)) % Check if tperf has non-NaN values
                plot(tr.epoch, tr.tperf, 'g-.', 'LineWidth', 1.5);
                legendEntries{end+1} = 'Test';
            end
            title('Training Performance'); xlabel('Epoch'); ylabel(upper(net.performFcn));
            legend(legendEntries, 'Location', 'northeast');
            hold off;
    
            subplot(3,1,2); cla; % Clear previous plot
            plot(tr.epoch, tr.gradient); title('Gradient Magnitude'); xlabel('Epoch'); ylabel('Gradient');
            
            subplot(3,1,3); cla; % Clear previous plot
            if validation && length(tr.epoch) > 1
                plot(tr.epoch(2:end), diff(tr.vperf)); title('\Delta Validation Error'); xlabel('Epoch'); ylabel('\Delta Error');
            elseif length(tr.epoch) > 1
                plot(tr.epoch(2:end), diff(tr.perf)); title('\Delta Training Error'); xlabel('Epoch'); ylabel('\Delta Error');
            else
                 title('\Delta Error (Requires >1 epoch)'); xlabel('Epoch'); ylabel('\Delta Error');
            end
            drawnow;
        end
        
        % Re-assign pdW, pdb; (crucial)
        pdW = dW;
        pdb = db;
        
        % Update best train performance.
        if Etrain < best_perf
            best_perf = Etrain;
        end
        % Store current best performance at each epoch (simplifies logic)
        tr.best_perf = [tr.best_perf, best_perf]; 
        
        % Update best validation performance.
        if validation
            if Eval < best_vperf
                best_vperf = Eval;
                best_net = net;
                best_epoch = epoch;
                val_fail = 0;
            else
                val_fail = val_fail + 1;
            end
            % Store current best validation performance and best epoch
            tr.best_vperf = [tr.best_vperf, best_vperf];
            tr.best_epoch = [tr.best_epoch, best_epoch];
        else
            % If no validation, keep track of best training performance epoch
             if Etrain <= best_perf % Use <= to capture the first time best is hit
                 best_epoch = epoch; % Update best epoch based on training perf
                 best_net = net;     % Update best network based on training perf
             end
             tr.best_epoch = [tr.best_epoch, best_epoch]; % Store current best epoch
        end
        
        % Update best test performance.
        if testing && ~isnan(Etest) % Only update if Etest is valid
            if Etest < best_tperf
                best_tperf = Etest;
            end
            tr.best_tperf = [tr.best_tperf, best_tperf];
        elseif testing % If testing but Etest is NaN
             tr.best_tperf = [tr.best_tperf, best_tperf]; % Keep the previous best
        end
        
        tr.val_fail = [tr.val_fail, val_fail];
        
        % --- Stop Criteria Checks ---
        stopReason = ''; % Initialize empty stop reason for this epoch
        if validation && val_fail >= max_fail
            stopReason = 'Maximum validation failures reached.';
        elseif Etrain <= goal % Check training performance against goal
            stopReason = 'Performance goal met.';
        elseif epoch >= epochs
            stopReason = 'Maximum epoch reached.';
        elseif gradient <= min_grad % Check minimum gradient (already checked earlier, but good to have here too)
            stopReason = 'Minimum gradient reached.';
        elseif ~isempty(tr.stop) && ~strcmp(tr.stop{end}, 'User closed the GUI window.') % Check if user closed window
            stopReason = tr.stop{end}; % Propagate existing stop reason
        end
    
        if ~isempty(stopReason)
            tr.stop = [tr.stop, {stopReason}];
            fprintf('\nINFO: Stopping training. Reason: %s\n', stopReason);
            break; % Exit the while loop
        end
    
        % --- Command Line Output ---
        % Simplified output to reduce verbosity; reinstate if needed
        if mod(epoch, net.trainParam.show) == 0 
            fprintf('Epoch: %d/%d (%.1f%%), TrainMSE: %.6f', epoch, epochs, epoch/epochs*100, Etrain);
            if validation
                fprintf(', ValMSE: %.6f (Best: %.6f, Fail: %d)', Eval, best_vperf, val_fail);
            end
            if testing && ~isnan(Etest)
                fprintf(', TestMSE: %.6f (Best: %.6f)', Etest, best_tperf);
            elseif testing
                fprintf(', TestMSE: NaN');
            end
            fprintf(', Grad: %.6f\n', gradient);
        end
    
    end % End while loop
    
    % Final assignments after loop
    tr.num_epochs = epoch; % Record the actual number of epochs run
    net = best_net; % Return the best network found (based on validation or training perf)
    
    % Ensure final fields have consistent lengths if training stopped early
    final_len = length(tr.epoch);
    tr.perf = tr.perf(1:final_len);
    tr.gradient = tr.gradient(1:final_len);
    if validation, tr.vperf = tr.vperf(1:final_len); end
    if testing, tr.tperf = tr.tperf(1:final_len); end
    tr.best_perf = tr.best_perf(1:final_len);
    if validation, tr.best_vperf = tr.best_vperf(1:final_len); end
    if testing, tr.best_tperf = tr.best_tperf(1:final_len); end
    tr.best_epoch = tr.best_epoch(1:final_len);
    tr.val_fail = tr.val_fail(1:final_len);
    tr.trainInd = tr.trainInd(1:final_len);
    tr.valInd = tr.valInd(1:final_len);
    tr.testInd = tr.testInd(1:final_len);
    tr.time = tr.time(1:final_len);
    
    % Assign the final best values to the main struct fields for convenience
    % Use the last value recorded in the vectors
    tr.best_perf = tr.best_perf(end); 
    if validation, tr.best_vperf = tr.best_vperf(end); else tr.best_vperf = NaN; end
    if testing, tr.best_tperf = tr.best_tperf(end); else tr.best_tperf = NaN; end
    tr.best_epoch = tr.best_epoch(end); % Epoch where best net was saved

end % End nntrain function
    
%% Helper function to evaluate performance
function perf = evaluate_network(performFcn, error)
    % Helper function to evaluate performance based on error
    switch lower(performFcn)
        case 'mse'
            % Calculate MSE, handle potential empty error matrix (e.g., if no test samples)
            if isempty(error)
                perf = NaN;
            else
                perf = mean(mean(error.^2));
            end
        case 'sse'
            if isempty(error)
                perf = NaN;
            else
                perf = sum(sum(error.^2));
            end
        case 'mae'
             if isempty(error)
                perf = NaN;
            else
                perf = mean(mean(abs(error)));
             end
        case 'sae'
            if isempty(error)
                perf = NaN;
            else
                perf = sum(sum(abs(error)));
            end
        otherwise
            warning('NN:Performance:unknown', 'Unknown performance function: %s. Returning NaN.', performFcn);
            perf = NaN; % Return NaN for unknown function
    end
end