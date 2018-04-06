import mock
import textwrap
import unittest

from lizard import hardware_discovery


class TestHardwareDiscovery(unittest.TestCase):
    """test hardware discovery system"""

    @mock.patch("lizard.util.subp")
    def test_check_cpus(self, mock_subp):
        """test hardware_discovery.check_cpus"""
        lscpu_responses = (
            textwrap.dedent("""
            Architecture:        x86_64
            CPU op-mode(s):      32-bit, 64-bit
            Byte Order:          Little Endian
            CPU(s):              6
            On-line CPU(s) list: 0-5
            Thread(s) per core:  2
            Core(s) per socket:  3
            Socket(s):           1
            NUMA node(s):        1
            Vendor ID:           AuthenticAMD
            CPU family:          21
            Model:               2
            Model name:          AMD FX(tm)-6300 Six-Core Processor
            Stepping:            0
            CPU MHz:             1400.000
            CPU max MHz:         3700.0000
            CPU min MHz:         1400.0000
            BogoMIPS:            7424.81
            Virtualization:      AMD-V
            L1d cache:           16K
            L1i cache:           64K
            L2 cache:            2048K
            L3 cache:            8192K
            NUMA node0 CPU(s):   0-5
            Flags:               fpu vme de pse tsc msr pae mce cx8
            apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse
            sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm
            constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid
            aperfmperf pni pclmulqdq monitor ssse3 fma cx16 sse4_1
            sse4_2 popcnt aes xsave avx f16c lahf_lm cmp_legacy svm
            extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw
            ibs xop skinit wdt lwp fma4 tce nodeid_msr tbm topoext
            perfctr_core perfctr_nb cpb hw_pstate retpoline
            retpoline_amd rsb_ctxsw vmmcall bmi1 arat npt lbrv svm_lock
            nrip_save tsc_scale vmcb_clean flushbyasid decodeassists
            pausefilter pfthreshold"""),
            textwrap.dedent("""
            Architecture:          x86_64
            CPU op-mode(s):        32-bit, 64-bit
            Byte Order:            Little Endian
            CPU(s):                8
            On-line CPU(s) list:   0-7
            Thread(s) per core:    1
            Core(s) per socket:    4
            Socket(s):             2
            NUMA node(s):          2
            Vendor ID:             GenuineIntel
            CPU family:            6
            Model:                 62
            Model name:            Intel(R) Xeon(R) CPU E5-2603 v2 @ 1.80GHz
            Stepping:              4
            CPU MHz:               1279.265
            CPU max MHz:           1800.0000
            CPU min MHz:           1200.0000
            BogoMIPS:              3601.63
            Virtualization:        VT-x
            L1d cache:             32K
            L1i cache:             32K
            L2 cache:              256K
            L3 cache:              10240K
            NUMA node0 CPU(s):     0,2,4,6
            NUMA node1 CPU(s):     1,3,5,7
            Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep
            mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss
            ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon
            pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu
            pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr
            pdcm pcid dca sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes
            xsave avx f16c rdrand lahf_lm kaiser tpr_shadow vnmi flexpriority
            ept vpid fsgsbase smep erms xsaveopt dtherm arat pln pts""")
        )
        expected_responses = (
            {'max_threads': 6,
             'name': 'AMD FX(tm)-6300 Six-Core Processor'},
            {'max_threads': 8,
             'name': 'Intel(R) Xeon(R) CPU E5-2603 v2 @ 1.80GHz'}
        )
        for lscpu_output, expected in zip(lscpu_responses, expected_responses):
            mock_subp.return_value = (lscpu_output, "")
            self.assertEqual(hardware_discovery.check_cpus(), expected)
            mock_subp.assert_called_with(["lscpu"])
