// +build amd64

// func DotProduct(left, right []float32, result float32) float32
TEXT ·DotProduct(SB), 4, $0
    // Load slice lengths
    MOVQ    leftLen+8(FP), AX
    MOVQ    rightLen+32(FP), BX
    
    // Get minimum length
    CMPQ    AX, BX
    CMOVQLT AX, BX
    
    // Load slice data pointers
    MOVQ    leftData+0(FP), SI
    MOVQ    rightData+24(FP), DX
    
    // Initialize accumulator registers
    VXORPS  Y2, Y2, Y2  // Y2 will hold partial sums for vector operations
    XORPS   X3, X3      // X3 will hold the final sum
    
    // Initialize loop index
    MOVQ    $0, CX
vectorLoop:
    MOVQ    BX, DI
    SUBQ    CX, DI
    CMPQ    DI, $8
    JL      singleLoop
    
    // Process 8 float32 values at once
    VMOVUPS (SI)(CX*4), Y0
    VMOVUPS (DX)(CX*4), Y1
    VMULPS  Y0, Y1, Y0  // Multiply vectors
    VADDPS  Y0, Y2, Y2  // Add to accumulator
    
    ADDQ    $8, CX
    JMP     vectorLoop

singleLoop:
    CMPQ    CX, BX
    JGE     reduction
    
    // Process one float32 value
    MOVSS   (SI)(CX*4), X0
    MOVSS   (DX)(CX*4), X1
    MULSS   X1, X0
    ADDSS   X0, X3
    
    INCQ    CX
    JMP     singleLoop

reduction:
    // Reduce Y2 vector register to scalar
    VEXTRACTF128 $1, Y2, X1
    VEXTRACTF128 $0, Y2, X0
    ADDPS   X1, X0
    HADDPS  X0, X0
    HADDPS  X0, X0
    
    // Add the vector sum to the scalar sum
    ADDSS   X0, X3
    
    // Add input result value
    MOVSS   result+48(FP), X0
    ADDSS   X0, X3
    
    // Store final result
    MOVSS   X3, ret+56(FP)
    RET
