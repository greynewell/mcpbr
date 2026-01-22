#!/usr/bin/env node

/**
 * mcpbr CLI wrapper for npm
 *
 * This wrapper provides npm/npx access to the mcpbr CLI tool,
 * which is implemented in Python. It checks for Python 3.11+
 * and the mcpbr Python package, then forwards all arguments
 * to the Python CLI.
 */

const { spawn } = require('cross-spawn');
const { execSync } = require('child_process');

/**
 * Check if Python 3.11+ is available
 */
function checkPython() {
  try {
    // Try python3 first (most common on Unix systems)
    const version = execSync('python3 --version', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
    const match = version.match(/Python (\d+)\.(\d+)/);

    if (match) {
      const major = parseInt(match[1]);
      const minor = parseInt(match[2]);

      if (major === 3 && minor >= 11) {
        return 'python3';
      }
    }
  } catch (error) {
    // python3 not found or failed
  }

  try {
    // Try python (Windows common, sometimes Unix too)
    const version = execSync('python --version', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
    const match = version.match(/Python (\d+)\.(\d+)/);

    if (match) {
      const major = parseInt(match[1]);
      const minor = parseInt(match[2]);

      if (major === 3 && minor >= 11) {
        return 'python';
      }
    }
  } catch (error) {
    // python not found or failed
  }

  return null;
}

/**
 * Check if mcpbr Python package is installed
 */
function checkMcpbr(pythonCmd) {
  try {
    execSync(`${pythonCmd} -m mcpbr --version`, {
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'ignore']
    });
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Print installation instructions
 */
function printInstallInstructions() {
  console.error(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mcpbr requires Python 3.11+ and the mcpbr Python package
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please install the requirements:

  1. Install Python 3.11 or later:
     • macOS: brew install python@3.11
     • Ubuntu: sudo apt install python3.11
     • Windows: https://www.python.org/downloads/

  2. Install mcpbr via pip:
     • pip install mcpbr
     or
     • pip3 install mcpbr

For more information, visit: https://github.com/greynewell/mcpbr

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
}

/**
 * Print Python version mismatch error
 */
function printPythonVersionError() {
  console.error(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mcpbr requires Python 3.11 or later
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your Python version is too old. Please upgrade:

  • macOS: brew install python@3.11
  • Ubuntu: sudo apt install python3.11
  • Windows: https://www.python.org/downloads/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
}

/**
 * Print mcpbr not installed error
 */
function printMcpbrNotInstalledError() {
  console.error(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mcpbr Python package not found
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please install mcpbr via pip:

  pip install mcpbr

or

  pip3 install mcpbr

For more information, visit: https://github.com/greynewell/mcpbr

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
}

/**
 * Main execution
 */
function main() {
  // Check for Python 3.11+
  const pythonCmd = checkPython();

  if (!pythonCmd) {
    printPythonVersionError();
    process.exit(1);
  }

  // Check if mcpbr is installed
  if (!checkMcpbr(pythonCmd)) {
    printMcpbrNotInstalledError();
    process.exit(1);
  }

  // Forward all arguments to mcpbr Python CLI
  const args = process.argv.slice(2);
  const mcpbr = spawn(pythonCmd, ['-m', 'mcpbr', ...args], {
    stdio: 'inherit',
    env: process.env
  });

  mcpbr.on('error', (error) => {
    console.error(`Failed to start mcpbr: ${error.message}`);
    process.exit(1);
  });

  mcpbr.on('exit', (code, signal) => {
    // If killed by signal, exit with error code
    if (signal) {
      process.exit(1);
    }
    process.exit(code || 0);
  });
}

// Run if executed directly
if (require.main === module) {
  main();
}

module.exports = { checkPython, checkMcpbr };
