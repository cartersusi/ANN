// +build amd64

// func Supported() bool
TEXT ·Supported(SB), 4, $0
    //Check XSAVE supported.
    MOVQ    $1, AX
    CPUID
    TESTQ   $(1<<26), CX
    JZ      avxFalse
    //Check AVX supported.
    TESTQ   $(1<<28), CX
    JZ      avxFalse
    //Check XMM and YMM enabled.
    MOVQ    $0, CX
    XGETBV
    TESTQ   $(1<<1), AX
    JZ      avxFalse
    TESTQ   $(1<<2), AX
    JZ      avxFalse
    //avxTrue:
    MOVB    $1, bool+0(FP)
    RET
avxFalse:
    MOVB    $0, bool+0(FP)
    RET
