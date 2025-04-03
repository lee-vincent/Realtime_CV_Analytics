# trt_minimal_test.py
import tensorrt as trt
import sys
import os
import traceback # Import traceback for detailed error printing

# --- Configuration ---
# !! MODIFY THESE AS NEEDED !!
VENV_REL_PATH = 'dnn_env' # Relative path to venv from script location
ENGINE_REL_PATH = 'yolov5/yolov5s.trt' # Relative path to engine from script location
# !! --------------------- !!

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct absolute paths
VENV_BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, VENV_REL_PATH))
VENV_SITE_PACKAGES = os.path.abspath(os.path.join(VENV_BASE_DIR, 'lib', 'site-packages'))
ENGINE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, ENGINE_REL_PATH))


print(f"--- Python Executable: {sys.executable} ---")
print(f"--- Script Directory: {SCRIPT_DIR} ---")
print(f"--- Expected Venv Site Packages: {VENV_SITE_PACKAGES} ---")
print(f"--- Using Engine File: {ENGINE_PATH} ---")


# --- Step 0: Check prerequisites and modify sys.path ---
print("\n--- Step 0: Prerequisites and sys.path ---")
if not os.path.exists(VENV_SITE_PACKAGES):
    print(f"WARNING: Expected venv site-packages directory not found at '{VENV_SITE_PACKAGES}'. Path modification skipped.")
elif sys.path[0] != VENV_SITE_PACKAGES:
    print(f"Current sys.path[0]: {sys.path[0] if sys.path else 'None'}")
    # Remove it if it exists elsewhere in the path to avoid duplicates
    while VENV_SITE_PACKAGES in sys.path:
        sys.path.remove(VENV_SITE_PACKAGES)
    # Insert it at the beginning
    sys.path.insert(0, VENV_SITE_PACKAGES)
    print(f"INFO: Inserted '{VENV_SITE_PACKAGES}' at the start of sys.path")
else:
     print("INFO: Venv site-packages already at start of sys.path.")

print(f"Final sys.path: {sys.path}")

if not os.path.exists(ENGINE_PATH):
    print(f"FATAL ERROR: Engine file not found at '{ENGINE_PATH}'")
    sys.exit(1) # Exit if engine file is missing
# -------------------------------------------------------------


# --- Step 1: Report TensorRT Module Info ---
print("\n--- Step 1: TensorRT Module Info ---")
try:
    print(f"Location: {trt.__file__}")
except AttributeError:
    print("Location: Could not determine trt.__file__")
# Get version using official API preferably
try:
    version_info = trt.getVersion()
    trt_version_str = f"{version_info[0]}.{version_info[1]}.{version_info[2]}"
    print(f"Version (trt.getVersion()): {trt_version_str}")
except AttributeError:
    # Fallback if getVersion isn't available (older TRT?)
     try:
        print(f"Version (trt.__version__): {trt.__version__}")
     except AttributeError:
        print("Version: Could not determine TensorRT version.")
# -------------------------------------------------------------


print("\n--- Step 2: Starting Minimal TensorRT Object Creation ---")

# Initialize variables for cleanup
logger = None
runtime = None
engine = None
context = None

try:
    # 2.1 Create Logger
    print("Creating Logger...")
    logger = trt.Logger(trt.Logger.WARNING)
    if not isinstance(logger, trt.ILogger): # Check type
        raise TypeError(f"Failed: Logger is not of type trt.ILogger (got {type(logger)})")
    print("  Logger OK")

    # 2.2 Create Runtime
    print("Creating Runtime...")
    runtime = trt.Runtime(logger)
    if not isinstance(runtime, trt.Runtime): # Check type
        raise TypeError(f"Failed: Runtime is not of type trt.Runtime (got {type(runtime)})")
    print("  Runtime OK")

    # 2.3 Load Engine
    print("Loading Engine...")
    with open(ENGINE_PATH, "rb") as f:
        engine_bytes = f.read()
        if not engine_bytes: raise ValueError("Failed: Engine file is empty or could not be read")
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    if not isinstance(engine, trt.ICudaEngine): # Check type
        # The error message from TRT is usually printed by the logger,
        # but we add a clear failure indication here.
        raise RuntimeError("Failed: deserialize_cuda_engine did not return a valid ICudaEngine object.")
    print("  Engine OK")

    # 2.4 Create Execution Context
    print("Creating Execution Context...")
    # Try to ensure CUDA context exists, helpful if PyCUDA isn't explicitly used elsewhere
    try:
        # Attempt to import pycuda.autoinit only if pycuda is installed
        # This avoids forcing a dependency if not needed, but initializes context if available
        import importlib
        if importlib.util.find_spec("pycuda") is not None:
             import pycuda.autoinit
             print("  (pycuda.autoinit imported successfully - CUDA context likely initialized)")
        else:
             print("  (pycuda not found, context creation relies on TRT/CUDA runtime)")
    except Exception as e:
        print(f"  (Error importing pycuda.autoinit: {e}, continuing...)")

    context = engine.create_execution_context()
    if not isinstance(context, trt.IExecutionContext): # Check type
        raise TypeError(f"Failed: Context is not of type trt.IExecutionContext (got {type(context)})")
    print("  Context OK")

    # -------------------------------------------------------------

    # --- Step 3: Check Context Object and Methods ---
    print("\n--- Step 3: Context Object Analysis ---")
    print(f"Context type: {type(context)}")
    print(f"Context repr(): {repr(context)}") # Get representation of the object

    print("\nChecking for 'execute_async_v2' method...")
    has_method_v2 = hasattr(context, 'execute_async_v2')
    print(f"  hasattr(context, 'execute_async_v2'): {has_method_v2}")

    # Try direct access if hasattr reported True
    if has_method_v2:
        try:
            method_ref = context.execute_async_v2
            print(f"  Direct access context.execute_async_v2: SUCCEEDED, Method Type: {type(method_ref)}")
        except AttributeError:
             print(f"  Direct access context.execute_async_v2: FAILED with AttributeError (Contradicts hasattr!)") # Should not happen

    print("\nChecking for 'execute_async' method...")
    has_method_v1 = hasattr(context, 'execute_async')
    print(f"  hasattr(context, 'execute_async'): {has_method_v1}")

    # Try direct access if hasattr reported True
    if has_method_v1:
        try:
            method_ref = context.execute_async
            print(f"  Direct access context.execute_async: SUCCEEDED, Method Type: {type(method_ref)}")
        except AttributeError:
             print(f"  Direct access context.execute_async: FAILED with AttributeError (Contradicts hasattr!)")

    print("\nChecking for 'execute_v2' (synchronous) method...") # ADDED THIS CHECK
    has_method_sync_v2 = hasattr(context, 'execute_v2')
    print(f"  hasattr(context, 'execute_v2'): {has_method_sync_v2}")

    # Try direct access if hasattr reported True
    if has_method_sync_v2:
        try:
            method_ref = context.execute_v2
            print(f"  Direct access context.execute_v2: SUCCEEDED, Method Type: {type(method_ref)}")
        except AttributeError:
             print(f"  Direct access context.execute_v2: FAILED with AttributeError (Contradicts hasattr!)")
    # -------------------------------------------------------------

    # --- Step 4: Final Conclusion ---
    print("\n--- Step 4: Test Conclusion ---")
    if not has_method_v2:
         print("---> RESULT: FAILURE - 'execute_async_v2' method was NOT found.")
    else:
         print("---> RESULT: SUCCESS - 'execute_async_v2' method WAS found.")

    # Add notes about the other methods based on the checks
    if not has_method_v1:
         print("---> NOTE: 'execute_async' (older API) method was ALSO NOT found.")
    else:
         print("---> NOTE: 'execute_async' (older API) method WAS found.")

    if not has_method_sync_v2:
         print("---> NOTE: 'execute_v2' (synchronous API) method was ALSO NOT found.")
    else:
         print("---> NOTE: 'execute_v2' (synchronous API) method WAS found.")
    # -------------------------------------------------------------


except Exception as e:
    print(f"\n--- AN ERROR OCCURRED DURING MINIMAL TEST ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    print("\nFull Traceback:")
    traceback.print_exc() # Print the full error traceback

finally:
    # --- Step 5: Cleanup ---
    print("\n--- Step 5: Cleanup ---")
    # Delete in reverse order of creation, checking existence first
    # Using locals() to check if variable was defined in the try block
    if 'context' in locals() and context is not None:
        print("Deleting context...")
        del context
    if 'engine' in locals() and engine is not None:
        print("Deleting engine...")
        del engine
    if 'runtime' in locals() and runtime is not None:
        print("Deleting runtime...")
        del runtime
    if 'logger' in locals() and logger is not None:
        print("Deleting logger...")
        del logger
    print("Minimal test finished.")
    # -------------------------------------------------------------