/*
 * COMP 321 Project 3: Malloc
 *
 * Simple, 32-bit and 64-bit clean allocator based on an implicit free list,
 * first fit placement, and boundary tag coalescing, as described in the
 * CS:APP3e text.  Blocks are aligned to double-word boundaries.  This
 * yields 8-byte aligned blocks on a 32-bit processor, and 16-byte aligned
 * blocks on a 64-bit processor.  However, 16-byte alignment is stricter
 * than necessary; the assignment only requires 8-byte alignment.  The
 * minimum block size is four words.
 *
 * This allocator uses the size of a pointer, e.g., sizeof(void *), to
 * define the size of a word.  This allocator also uses the standard
 * type uintptr_t to define unsigned integers that are the same size
 * as a pointer, i.e., sizeof(uintptr_t) == sizeof(void *).
 *
 * <Put your name and NetID here>
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "memlib.h"
#include "mm.h"

/* Basic constants and macros: */
#define WSIZE	  sizeof(void *) /* Word and header/footer size (bytes) */
#define DSIZE	  (2 * WSIZE)	 /* Doubleword size (bytes) */
#define CHUNKSIZE (1 << 12)	 /* Extend heap by this amount (bytes) */

#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* Pack a size and allocated bit into a word. */
#define PACK(size, alloc) ((size) | (alloc))

/* Read and write a word at address p. */
#define GET(p)	    (*(uintptr_t *)(p))
#define PUT(p, val) (*(uintptr_t *)(p) = (val))

/* Read the size and allocated fields from address p. */
#define GET_SIZE(p)  (GET(p) & ~(DSIZE - 1))
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer. */
#define HDRP(bp) ((char *)(bp)-WSIZE)
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* Given block ptr bp, compute address of next and previous blocks. */
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(((char *)(bp)-WSIZE)))
#define PREV_BLKP(bp) ((char *)(bp)-GET_SIZE(((char *)(bp)-DSIZE)))

/* Global variables: */
static char *heap_listp; /* Pointer to first block */

/* Function prototypes for internal helper routines: */
static void *coalesce(void *bp);
static void *extend_heap(size_t words);
static void *find_fit(size_t asize);
static void place(void *bp, size_t asize);

/* Function prototypes for heap consistency checker routines: */
static void checkblock(void *bp);
static void checkheap(bool verbose);
static void printblock(void *bp);

/*
 * Initialize the memory manager.
 *
 * @return 0 if the memory manager was successfully initialized and
 *   -1 otherwise.
 */
int
mm_init(void)
{

	/* Create the initial empty heap. */
	if ((heap_listp = mem_sbrk(4 * WSIZE)) == (void *)-1)
		return (-1);
	PUT(heap_listp, 0);			       /* Alignment padding */
	PUT(heap_listp + (1 * WSIZE), PACK(DSIZE, 1)); /* Prologue header */
	PUT(heap_listp + (2 * WSIZE), PACK(DSIZE, 1)); /* Prologue footer */
	PUT(heap_listp + (3 * WSIZE), PACK(0, 1));     /* Epilogue header */
	heap_listp += (2 * WSIZE);

	/* Extend the empty heap with a free block of CHUNKSIZE bytes. */
	if (extend_heap(CHUNKSIZE / WSIZE) == NULL)
		return (-1);
	return (0);
}

/*
 * Allocate a block with at least "size" bytes of payload, unless "size" is
 * zero.
 *
 * @param size the minimum size of the block's payload area.
 * @return the address of this block if the allocation was successful
 *   and NULL otherwise.
 */
void *
mm_malloc(size_t size)
{
	size_t asize;	   /* Adjusted block size */
	size_t extendsize; /* Amount to extend heap if no fit */
	void *bp;

	/* Ignore spurious requests. */
	if (size == 0)
		return (NULL);

	/* Adjust block size to include overhead and alignment reqs. */
	if (size <= DSIZE)
		asize = 2 * DSIZE;
	else
		asize = DSIZE * ((size + DSIZE + (DSIZE - 1)) / DSIZE);

	/* Search the free list for a fit. */
	if ((bp = find_fit(asize)) != NULL) {
		place(bp, asize);
		return (bp);
	}

	/* No fit found.  Get more memory and place the block. */
	extendsize = MAX(asize, CHUNKSIZE);
	if ((bp = extend_heap(extendsize / WSIZE)) == NULL)
		return (NULL);
	place(bp, asize);
	return (bp);
}

/*
 * Free a block.
 *
 * @param bp either the address of an allocated block or NULL.
 */
void
mm_free(void *bp)
{
	size_t size;

	/* Ignore spurious requests. */
	if (bp == NULL)
		return;

	/* Free and coalesce the block. */
	size = GET_SIZE(HDRP(bp));
	PUT(HDRP(bp), PACK(size, 0));
	PUT(FTRP(bp), PACK(size, 0));
	coalesce(bp);
}

/*
 * Reallocates the block "ptr" to a block with at least "size" bytes of
 * payload, unless "size" is zero.  If "size" is zero, frees the block
 * "ptr" and returns NULL.  If the block "ptr" is already a block with at
 * least "size" bytes of payload, then "ptr" may optionally be returned.
 * Otherwise, a new block is allocated and the contents of the old block
 * "ptr" are copied to that new block.  Returns the address of this new
 * block if the allocation was successful and NULL otherwise.
 *
 * @param ptr either the address of an allocated block or NULL.
 * @return the address of this block if the reallocation was successful
 *   and NULL otherwise.
 */
void *
mm_realloc(void *ptr, size_t size)
{
	size_t oldsize;
	void *newptr;

	/* If size == 0 then this is just free, and we return NULL. */
	if (size == 0) {
		mm_free(ptr);
		return (NULL);
	}

	/* If oldptr is NULL, then this is just malloc. */
	if (ptr == NULL)
		return (mm_malloc(size));

	newptr = mm_malloc(size);

	/* If realloc() fails, the original block is left untouched.  */
	if (newptr == NULL)
		return (NULL);

	/* Copy just the old data, not the old header and footer. */
	oldsize = GET_SIZE(HDRP(ptr)) - DSIZE;
	if (size < oldsize)
		oldsize = size;
	memcpy(newptr, ptr, oldsize);

	/* Free the old block. */
	mm_free(ptr);

	return (newptr);
}

/*
 * The following routines are internal helper routines.
 */

/*
 * Perform boundary tag coalescing.
 *
 * @param bp the address of a newly freed block.
 * @return the address of the coalesced block.
 */
static void *
coalesce(void *bp)
{
	size_t size = GET_SIZE(HDRP(bp));
	bool prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
	bool next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));

	if (prev_alloc && next_alloc) { /* Case 1 */
		return (bp);
	} else if (prev_alloc && !next_alloc) { /* Case 2 */
		size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
		PUT(HDRP(bp), PACK(size, 0));
		PUT(FTRP(bp), PACK(size, 0));
	} else if (!prev_alloc && next_alloc) { /* Case 3 */
		size += GET_SIZE(HDRP(PREV_BLKP(bp)));
		PUT(FTRP(bp), PACK(size, 0));
		PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
		bp = PREV_BLKP(bp);
	} else { /* Case 4 */
		size += GET_SIZE(HDRP(PREV_BLKP(bp))) +
		    GET_SIZE(FTRP(NEXT_BLKP(bp)));
		PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
		PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
		bp = PREV_BLKP(bp);
	}
	return (bp);
}

/*
 * Extend the heap with a free block and return that block's address.
 *
 * @param words the minimum number of words in the new free block.
 * @return the address of the new block.
 */
static void *
extend_heap(size_t words)
{
	size_t size;
	void *bp;

	/* Allocate an even number of words to maintain alignment. */
	size = (words % 2) ? (words + 1) * WSIZE : words * WSIZE;
	if ((bp = mem_sbrk(size)) == (void *)-1)
		return (NULL);

	/* Initialize free block header/footer and the epilogue header. */
	PUT(HDRP(bp), PACK(size, 0));	      /* Free block header */
	PUT(FTRP(bp), PACK(size, 0));	      /* Free block footer */
	PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1)); /* New epilogue header */

	/* Coalesce if the previous block was free. */
	return (coalesce(bp));
}

/*
 * Find a fit for a block with "asize" bytes.
 *
 * @param asize the minumum size of the returned block.
 * @return the block's address or NULL if no suitable block was found.
 */
static void *
find_fit(size_t asize)
{
	void *bp;

	/* Search for the first fit. */
	for (bp = heap_listp; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
		if (!GET_ALLOC(HDRP(bp)) && asize <= GET_SIZE(HDRP(bp)))
			return (bp);
	}
	/* No fit was found. */
	return (NULL);
}

/*
 * Place a block of "asize" bytes at the start of the free block "bp" and
 * split that block if the remainder would be at least the minimum block
 * size.
 *
 * @param bp the address of a free block
 * @param asize the minimum size of the placed block.
 */
static void
place(void *bp, size_t asize)
{
	size_t csize = GET_SIZE(HDRP(bp));

	if ((csize - asize) >= (2 * DSIZE)) {
		PUT(HDRP(bp), PACK(asize, 1));
		PUT(FTRP(bp), PACK(asize, 1));
		bp = NEXT_BLKP(bp);
		PUT(HDRP(bp), PACK(csize - asize, 0));
		PUT(FTRP(bp), PACK(csize - asize, 0));
	} else {
		PUT(HDRP(bp), PACK(csize, 1));
		PUT(FTRP(bp), PACK(csize, 1));
	}
}

/*
 * The remaining routines are heap consistency checker routines.
 */

/*
 * Perform a minimal check on the block "bp".
 *
 * @param bp the address of a block.
 */
static void
checkblock(void *bp)
{

	if ((uintptr_t)bp % DSIZE)
		printf("Error: %p is not doubleword aligned\n", bp);
	if (GET(HDRP(bp)) != GET(FTRP(bp)))
		printf("Error: header does not match footer\n");
}

/*
 * Perform a minimal check of the heap for consistency.
 *
 * @param verbose enable verbose debugging output.
 */
void
checkheap(bool verbose)
{
	void *bp;

	if (verbose)
		printf("Heap (%p):\n", heap_listp);

	if (GET_SIZE(HDRP(heap_listp)) != DSIZE || !GET_ALLOC(HDRP(heap_listp)))
		printf("Bad prologue header\n");
	checkblock(heap_listp);

	for (bp = heap_listp; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
		if (verbose)
			printblock(bp);
		checkblock(bp);
	}

	if (verbose)
		printblock(bp);
	if (GET_SIZE(HDRP(bp)) != 0 || !GET_ALLOC(HDRP(bp)))
		printf("Bad epilogue header\n");
}

/*
 * Print a block.
 *
 * @param bp the address of a block.
 */
static void
printblock(void *bp)
{
	size_t hsize, fsize;
	bool halloc, falloc;

	checkheap(false);
	hsize = GET_SIZE(HDRP(bp));
	halloc = GET_ALLOC(HDRP(bp));
	fsize = GET_SIZE(FTRP(bp));
	falloc = GET_ALLOC(FTRP(bp));

	if (hsize == 0) {
		printf("%p: end of heap\n", bp);
		return;
	}

	printf("%p: header: [%zu:%c] footer: [%zu:%c]\n", bp, hsize,
	    (halloc ? 'a' : 'f'), fsize, (falloc ? 'a' : 'f'));
}
