import numpy as np
from numba import cuda, float32, int32
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states

# ======================
# Parámetros del GA/GP
# ======================
POP = 1_024
GENE_LEN = 64
THREADS = 256
BLOCKS_POP = (POP + THREADS - 1) // THREADS
BLOCKS_GENES = ((POP * GENE_LEN) + THREADS - 1) // THREADS

# ======================
# Kernels (plantillas)
# ======================
@cuda.jit
def init_population(genes, rng_states):
    i = cuda.grid(1)
    if i < genes.size:
        # genes: [POP, GENE_LEN] aplanado row-major
        # Inicializa aleatorio (ajusta a tu alfabeto/operadores)
        r = xoroshiro128p_uniform_float32(rng_states, i)
        genes[i] = int32(r * 10)  # ejemplo

@cuda.jit
def evaluate(genes, fitness, problem_data):
    # Supón 1 hilo por individuo
    idx = cuda.grid(1)
    if idx < fitness.size:
        # Acceso a los genes del individuo idx:
        # base = idx * GENE_LEN
        # ... calcula fitness ...
        val = 0.0
        # Ejemplo tonto: suma de genes
        base = idx * GENE_LEN
        for j in range(GENE_LEN):
            val += float32(genes[base + j])
        fitness[idx] = val

@cuda.jit
def selection_tournament(genes_in, fitness, genes_out, rng_states, tour_k):
    idx = cuda.grid(1)
    if idx < (genes_out.size // GENE_LEN):
        # Torneo: elige mejor de k al azar
        best = -1
        best_fit = -1e30
        for t in range(tour_k):
            r = xoroshiro128p_uniform_float32(rng_states, idx*tour_k + t)
            cand = int32(r * fitness.size)
            f = fitness[cand]
            if f > best_fit:
                best_fit = f; best = cand
        # Copia genes del ganador a salida
        src = best * GENE_LEN
        dst = idx * GENE_LEN
        for j in range(GENE_LEN):
            genes_out[dst + j] = genes_in[src + j]

@cuda.jit
def crossover_mutation(parents, offspring, rng_states, p_mut, p_xover):
    idx = cuda.grid(1)
    if idx < (offspring.size // GENE_LEN):
        # Simple 1-point crossover entre idx y idx^1 (parea 0-1, 2-3, ...)
        mate = idx ^ 1
        cut = int32(xoroshiro128p_uniform_float32(rng_states, idx) * GENE_LEN)
        dst = idx * GENE_LEN
        srcA = idx * GENE_LEN
        srcB = (mate % (offspring.size // GENE_LEN)) * GENE_LEN

        # crossover
        for j in range(GENE_LEN):
            if j < cut:
                offspring[dst + j] = parents[srcA + j]
            else:
                offspring[dst + j] = parents[srcB + j]

        # mutation (ejemplo bit-flip / pequeño ajuste)
        for j in range(GENE_LEN):
            r = xoroshiro128p_uniform_float32(rng_states, idx * GENE_LEN + j)
            if r < p_mut:
                offspring[dst + j] ^= 1  # ejemplo

@cuda.jit
def reduce_argmax(values, out_max, out_idx):
    # Reducción por bloque + atómica simple (para claridad)
    smem_val = cuda.shared.array(256, dtype=float32)
    smem_idx = cuda.shared.array(256, dtype=int32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    v = -1e30
    idv = -1
    if i < values.size:
        v = values[i]; idv = i
    smem_val[tid] = v
    smem_idx[tid] = idv
    cuda.syncthreads()

    # reduce en el bloque
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            if smem_val[tid + s] > smem_val[tid]:
                smem_val[tid] = smem_val[tid + s]
                smem_idx[tid] = smem_idx[tid + s]
        s //= 2
        cuda.syncthreads()

    # escribe ganador del bloque
    if tid == 0:
        # usa atomics con un buffer global de tamaño 1 (simplificado)
        # Para producción, implementa una segunda pasada o usa cub-like
        if smem_val[0] > out_max[0]:
            out_max[0] = smem_val[0]
            out_idx[0] = smem_idx[0]

# ======================
# Setup de memoria
# ======================
def setup(pop=POP, gene_len=GENE_LEN, seed=1234):
    # Device (aplanado SoA simple)
    genes_A = cuda.device_array(pop * gene_len, dtype=np.int32)
    genes_B = cuda.device_array_like(genes_A)
    fitness = cuda.device_array(pop, dtype=np.float32)

    # RNG
    # Nota: usa número de hilos acorde a tus kernels (aquí para seguridad * gene_len)
    rng_states = create_xoroshiro128p_states(max(pop*gene_len, pop*THREADS), seed=seed)

    # Buffers de reducción (1 valor/índice en device)
    d_best_val = cuda.to_device(np.array([-1e30], dtype=np.float32))
    d_best_idx = cuda.to_device(np.array([-1], dtype=np.int32))

    # Host pinned para resúmenes
    h_best_val = cuda.pinned_array(1, dtype=np.float32)
    h_best_idx = cuda.pinned_array(1, dtype=np.int32)

    # Streams
    s_eval = cuda.stream()
    s_h2d = cuda.stream()
    s_d2h = cuda.stream()

    return {
        "genes_A": genes_A, "genes_B": genes_B, "fitness": fitness,
        "rng": rng_states, "d_best_val": d_best_val, "d_best_idx": d_best_idx,
        "h_best_val": h_best_val, "h_best_idx": h_best_idx,
        "s_eval": s_eval, "s_h2d": s_h2d, "s_d2h": s_d2h
    }

# ======================
# Bucle evolutivo
# ======================
def run_evolution(num_gens=200, p_mut=0.02, p_xover=0.9, tour_k=3):
    ctx = setup()
    gA = ctx["genes_A"]; gB = ctx["genes_B"]; fit = ctx["fitness"]
    rng = ctx["rng"]
    d_best_val, d_best_idx = ctx["d_best_val"], ctx["d_best_idx"]
    h_best_val, h_best_idx = ctx["h_best_val"], ctx["h_best_idx"]
    s_eval, s_h2d, s_d2h = ctx["s_eval"], ctx["s_h2d"], ctx["s_d2h"]

    # Inicialización (en GPU)
    init_population[BLOCKS_GENES, THREADS, s_eval](gA, rng)
    s_eval.synchronize()

    parents = gA
    offspring = gB

    for gen in range(num_gens):
        # 1) Evaluación
        evaluate[BLOCKS_POP, THREADS, s_eval](parents, fit, None)

        # 2) Reset reducción (mejor de la gen)
        d_best_val.copy_to_device(np.array([-1e30], dtype=np.float32), stream=s_h2d)
        d_best_idx.copy_to_device(np.array([-1], dtype=np.int32), stream=s_h2d)

        # 3) Reducción (argmax) en mismo stream de eval o independiente si separas fases
        #    (Aquí, simple: misma malla que evaluación)
        reduce_argmax[BLOCKS_POP, THREADS, s_eval](fit, d_best_val, d_best_idx)

        # 4) Baja resúmenes asíncronos
        d_best_val.copy_to_host(h_best_val, stream=s_d2h)
        d_best_idx.copy_to_host(h_best_idx, stream=s_d2h)

        # 5) Selección + variación (todo en GPU) usando parents -> temp -> offspring
        selection_tournament[BLOCKS_POP, THREADS, s_eval](parents, fit, offspring, rng, tour_k)
        crossover_mutation[BLOCKS_POP, THREADS, s_eval](offspring, offspring, rng, p_mut, p_xover)

        # 6) Sincroniza solo D→H para leer resultados en CPU y decidir paro/log
        s_d2h.synchronize()
        best_val = float(h_best_val[0]); best_idx = int(h_best_idx[0])

        # (Opcional) criterio de paro/logging aquí SIN sincronizar los demás streams
        # print(f"Gen {gen}: best {best_val} @ {best_idx}")

        # 7) Intercambia roles (ping-pong) sin copiar
        parents, offspring = offspring, parents

    # Asegura que todo terminó
    s_eval.synchronize()
    return best_val, best_idx
