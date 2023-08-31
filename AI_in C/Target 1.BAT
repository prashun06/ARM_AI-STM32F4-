SET PATH=C:\Keil_v5\ARM\ARMCLANG\Bin;C:\Program Files\Common Files\Oracle\Java\javapath;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\Python310\Scripts\;C:\Program Files\Python310\;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;C:\Users\adib13000\AppData\Local\Programs\Python\Python310;C:\Program Files\dotnet\;C:\Program Files (x86)\STMicroelectronics\STM32 ST-LINK Utility\ST-LINK Utility;C:\Program Files\Git\cmd;C:\Program Files (x86)\Kaspersky Lab\Endpoint Agent\;C:\Program Files\Polyspace\R2021a\runtime\win64;C:\Program Files\Polyspace\R2021a\bin;C:\Program Files\Polyspace\R2021a\polyspace\bin;C:\Program Files\PuTTY\;C:\Program Files\Microsoft SQL Server\Client SDK\ODBC\110\Tools\Binn\;C:\Program Files (x86)\Microsoft SQL Server\120\Tools\Binn\;C:\Program Files\Microsoft SQL Server\120\Tools\Binn\;C:\Program Files\Microsoft SQL Server\120\DTS\Binn\;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Users\user\AppData\Local\Programs\Python\Python311\Scripts\;C:\Users\user\AppData\Local\Programs\Python\Python311\;C:\Users\user\AppData\Local\Microsoft\WindowsApps;C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2023.1.3\bin;C:\Program Files\Common Files\Oracle\Java\javapath;;C:\Program Files\Common Files\Oracle\Java\javapath;
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
