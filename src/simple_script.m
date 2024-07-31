function [parameter1, parameter2, parameter3, arrayParameter] = loadProperties(filename)
    % loadProperties - Loads parameters from a JSON file.
    %
    % Syntax: [parameter1, parameter2, parameter3, arrayParameter, nestedParameter] = loadProperties(filename)
    %
    % Inputs:
    %    filename - String. The path to the JSON file.
    %
    % Outputs:
    %    parameter1 - Numeric. Example parameter.
    %    parameter2 - Numeric. Example parameter.
    %    parameter3 - String. Example parameter.
    %    arrayParameter - Array. Example array parameter.

    % Check if filename is provided
    if nargin < 1
        filename = 'properties.json'; % Default filename
    end

    % Read JSON file
    try
        jsonData = fileread(filename);  % Read the JSON file as a string
    catch
        error('Error reading the JSON file. Make sure the file exists and the path is correct.');
    end

    % Parse JSON data
    try
        data = jsondecode(jsonData);    % Decode JSON data into a MATLAB struct
    catch
        error('Error parsing JSON file. Ensure it is correctly formatted.');
    end

    % Extract parameters from the struct
    try
        parameter1 = data.parameter1;
        parameter2 = data.parameter2;
        parameter3 = data.parameter3;
        arrayParameter = data.arrayParameter;
    catch
        error('Error accessing JSON fields. Ensure JSON contains required fields.');
    end

    % Displaying the loaded parameters for debugging
    fprintf('Loaded parameter1: %f\n', parameter1);
    fprintf('Loaded parameter2: %f\n', parameter2);
    fprintf('Loaded parameter3: %s\n', parameter3);
    fprintf('Loaded arrayParameter: %s\n', mat2str(arrayParameter));
end

