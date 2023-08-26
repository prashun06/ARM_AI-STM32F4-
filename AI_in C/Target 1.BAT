SET PATH=C:\Keil_v5\ARM\ARMCLANG\Bin;C:\Python 3.11\Scripts\;C:\Python 3.11\;C:\Program Files\Common Files\Oracle\Java\javapath;C:\Program Files (x86)\Common Files\Oracle\Java\javapath;C:\Program Files (x86)\Common Files\Intel\Shared Libraries\redist\intel64\compiler;C:\Program Files (x86)\STMicroelectronics\st_toolset\asm;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\libnvvp;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\Program Files\Intel\WiFi\bin\;C:\Program Files\Common Files\Intel\WirelessCommon\;C:\Program Files\Notepad++;C:\Program Files\Polyspace\R2021a\runtime\win64;C:\Program Files\Polyspace\R2021a\bin;C:\Program Files\Polyspace\R2021a\polyspace\bin;C:\Program Files\PCL 1.8.0\bin;C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR;C:\Program Files\NVIDIA Corporation\Nsight Compute 2021.1.1\;C:\Program Files (x86)\STMicroelectronics\STM32 ST-LINK Utility\ST-LINK Utility;C:\Program Files\Microsoft SQL Server\Client SDK\ODBC\110\Tools\Binn\;C:\Program Files (x86)\Microsoft SQL Server\120\Tools\Binn\;C:\Program Files\Microsoft SQL Server\120\Tools\Binn\;C:\Program Files\Microsoft SQL Server\120\DTS\Binn\;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;C:\Program Files\PuTTY\;C:\MinGW\bin;C:\Program Files\Git\cmd;E:\Cadence\SPB_22.1\openaccess\bin\win32\opt;E:\Cadence\SPB_22.1\tools\capture;E:\Cadence\SPB_22.1\tools\pspice;E:\Cadence\SPB_22.1\tools\specctra\bin;E:\Cadence\SPB_22.1\tools\fet\bin;E:\Cadence\SPB_22.1\tools\libutil\bin;E:\Cadence\SPB_22.1\tools\bin;E:\Cadence\SPB_22.1\tools\pcb\bin;C:\Users\Asus\AppData\Local\Microsoft\WindowsApps;C:\Program Files\Intel\WiFi\bin\;C:\Program Files\Common Files\Intel\WirelessCommon\;C:\Program Files\JetBrains\PyCharm Community Edition 2021.1.1\bin;;C:\Users\Asus\AppData\Local\Programs\Microsoft VS Code\bin
SET ARMCC6_ASMOPT=--diag_suppress=9931
SET ARMCC6_CCOPT=--diag_suppress=9931
SET ARMCC6_LINKOPT=--diag_suppress=9931
SET CPU_TYPE=STM32F407VETx
SET CPU_VENDOR=STMicroelectronics
SET UV2_TARGET=Target 1
SET CPU_CLOCK=0x00B71B00
"C:\Keil_v5\ARM\ARMCLANG\Bin\ArmClang.exe" @".\objects\main.__i"
"C:\Keil_v5\ARM\ARMCLANG\Bin\ArmClang.exe" @".\objects\sin_in_sin_out.__i"
"C:\Keil_v5\ARM\ARMCLANG\Bin\ArmClang.exe" @".\objects\usart.__i"
"C:\Keil_v5\ARM\ARMCLANG\Bin\ArmAsm" --Via ".\objects\startup_stm32f407xx._ia"
"C:\Keil_v5\ARM\ARMCLANG\Bin\ArmClang.exe" @".\objects\system_stm32f4xx.__i"
"C:\Keil_v5\ARM\ARMCLANG\Bin\ArmLink" --Via ".\Objects\bluetooth.lnp"
"C:\Keil_v5\ARM\ARMCLANG\Bin\fromelf.exe" ".\Objects\bluetooth.axf" --i32combined --output ".\Objects\bluetooth.hex"
