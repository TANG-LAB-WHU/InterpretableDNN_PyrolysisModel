% Simple test script
disp('This is a test script to verify MATLAB execution');
disp(['Current directory: ' pwd]);
disp(['Script path: ' mfilename('fullpath')]);
disp(['MATLAB version: ' version]);

% Try to create a simple file
test_file = 'test_output.txt';
try
    fid = fopen(test_file, 'w');
    if fid == -1
        disp(['Error: Could not open file for writing: ' test_file]);
    else
        fprintf(fid, 'Test output file created at %s\n', datestr(now));
        fclose(fid);
        disp(['Successfully created test file: ' test_file]);
    end
catch ME
    disp(['Error writing file: ' ME.message]);
end

% Try to save a MAT file
try
    test_var = 'This is a test variable';
    save('test_save.mat', 'test_var');
    disp('Successfully saved test_save.mat');
catch ME
    disp(['Error saving MAT file: ' ME.message]);
end

disp('Test script completed'); 