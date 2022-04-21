# metro-maps
Use mixed integer programming to generate optimized metro-map layouts.

# Steps to run
*Requires Gurobi*
1. Specify input file path and model settings in optimizer.py
2. Run `python optimizer.py`

# Generating images
*Requires Node.js and npm*
*Requires powershell on windows*
1. Run `npm install` in the project directory
2. Run  `cat [input_path].json | npx svg-transit-map -y > [output_path]svg`.
   - Optional **Windows** script
     1. Specify the desired `.json` file and output file path in `generate` script within `package.json`.
     2. Run `npm run generate`
         
