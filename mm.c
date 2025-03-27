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
 * <Cindy Nguyen (cn32)>
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "memlib.h"
#include "mm.h"

/* Basic constants and macros: */
#define WSIZE	  sizeof(void *)      /* Word and header/footer size (bytes) */
#define DSIZE	  (2 * WSIZE)	      /* Doubleword size (bytes) */
#define CHUNKSIZE ((1 << 12) + DSIZE) /* Extend heap by this amount (bytes) */

#define ALIGNMENT 8 /* Alignment size (bytes). */
#define MIN_BLK_SIZE                                                      \
	(2 * DSIZE) /* Minimum block size (bytes): 1 DSIZE for header and \
		       footer, 1 DSIZE for prev and next pointers. */
#define NUM_BINS 6  /* Number of size bins. */

#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* Pack a size and allocated bit into a word. */
#define PACK(size, alloc) ((size) | (alloc))

/* Read and write a word at address p. */
#define GET(p)	    (*(uintptr_t *)(p))
#define PUT(p, val) (*(uintptr_t *)(p) = (val))

/* Read the size and allocated fields from address p. */
#define GET_SIZE(p)  (GET(p) & ~(ALIGNMENT - 1))
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer. */
#define HDRP(bp) ((char *)(bp) - WSIZE)
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* Given block ptr bp, compute address of next and previous blocks. */
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))
#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))

/* Define a struct containing pointers to the previous and next free blocks in
 * explicit free list. */
typedef struct fblock fblock;
struct fblock {
	struct fblock *prev;
	struct fblock *next;
};

/* Global variables: */
static char *heap_listp; /* Pointer to first block */

static fblock *flist; /* Pointer to dummy head of explicit free list. */

/* Function prototypes for internal helper routines: */
static void *coalesce(void *bp);
static void *extend_heap(size_t words);
static void *find_fit(size_t asize);
static void place(void *bp, size_t asize);

static void ins_flist(void *bp,
    size_t size); /* Function to insert a block into the segregated list
		     corresponding to its size in explicit free list. */
static void rem_flist(void *bp); /* Function to remove a block from the
				    segregated list in explicit free list. */
static fblock *size_bin(
    size_t size); /* Find the size bin corresponding to given size. */

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
	if ((heap_listp = mem_sbrk(NUM_BINS * DSIZE)) == (void *)-1)
		return (-1);
	/* Initialize explicit free list. */
	flist = (fblock *)heap_listp;

	/* Initialize size bin segregated lists. */
	for (int i = 0; i < NUM_BINS; i++) {
		fblock *bin_head = flist + i;
		bin_head->prev = bin_head;
		bin_head->next = bin_head;
	}
	heap_listp = (char *)(flist + NUM_BINS);

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
	void *bp; /* Pointer to the block to be returned for allocation. */

	/* Ignore spurious requests. */
	if (size == 0)
		return (NULL);

	/* Round the smaller requested sizes under 512 to the nearest larger
	 * power of 2 for efficiency. (approved) */
	if (size <= 16) {
		size = 16;
	} else if (size <= 32) {
		size = 32;
	} else if (size <= 64) {
		size = 64;
	} else if (size <= 128) {
		size = 128;
	} else if (size <= 256) {
		size = 256;
	} else if (size <= 512) {
		size = 512;
	}

	/* Adjust block size to include overhead and alignment reqs. */
	if (size <= DSIZE)
		asize = MIN_BLK_SIZE;
	else
		asize = ALIGNMENT *
		    ((size + DSIZE + (ALIGNMENT - 1)) / ALIGNMENT);

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
	size_t oldsize = GET_SIZE(HDRP(ptr)) - DSIZE;
	void *newptr;

	/* If size == 0 then this is just free, and we return NULL. */
	if (size == 0) {
		mm_free(ptr);
		return (NULL);
	}

	/* If size == oldsize then return ptr. */
	if (size == oldsize)
		return (ptr);

	/* If oldptr is NULL, then this is just malloc. */
	if (ptr == NULL)
		return (mm_malloc(size));

	/* Check if the previous and/or next block is free, if yes then maybe
	 * can coalesce with current block to make enough space for realloc
	 * size. */
	size_t csize = 0;
	newptr = ptr;

	/* If the realloc size is smaller then update oldsize. */
	if (size < oldsize) {
		csize = oldsize + DSIZE;
		oldsize = size;
	} else {
		bool prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(ptr)));
		bool next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(ptr)));

		if (prev_alloc && !next_alloc &&
		    (oldsize + GET_SIZE(HDRP(NEXT_BLKP(ptr))) >=
			size)) { /* Case 1: next block is free. */
			rem_flist(NEXT_BLKP(ptr));
			csize = oldsize + GET_SIZE(HDRP(NEXT_BLKP(ptr))) +
			    DSIZE;
		} else if (!prev_alloc && !next_alloc &&
		    (oldsize + GET_SIZE(HDRP(PREV_BLKP(ptr))) +
			    GET_SIZE(HDRP(NEXT_BLKP(ptr))) >=
			size)) { /* Case 1: previous and next blocks are free.
				  */
			rem_flist(PREV_BLKP(ptr));
			rem_flist(NEXT_BLKP(ptr));
			csize = oldsize + GET_SIZE(HDRP(PREV_BLKP(ptr))) +
			    GET_SIZE(HDRP(NEXT_BLKP(ptr))) + DSIZE;
			newptr = PREV_BLKP(ptr);
		}
	}

	if (csize > 0) {
		/* Only memcpy if not realloc in place. */
		if (newptr != ptr)
			memcpy(newptr, ptr, oldsize);

		/* Adjust block size to include overhead and alignment reqs. */
		size_t asize;
		if (size <= DSIZE)
			asize = MIN_BLK_SIZE;
		else
			asize = ALIGNMENT *
			    ((size + DSIZE + (ALIGNMENT - 1)) / ALIGNMENT);

		/* Split the block if the remianing size is at least minimum
		 * block size, otherwise allocate the whole block. */
		if ((csize - asize) >= (MIN_BLK_SIZE)) {
			PUT(HDRP(newptr), PACK(asize, 1));
			PUT(FTRP(newptr), PACK(asize, 1));
			PUT(HDRP(NEXT_BLKP(newptr)), PACK(csize - asize, 0));
			PUT(FTRP(NEXT_BLKP(newptr)), PACK(csize - asize, 0));
			ins_flist(NEXT_BLKP(newptr), csize - asize);
		} else {
			PUT(HDRP(newptr), PACK(csize, 1));
			PUT(FTRP(newptr), PACK(csize, 1));
		}
		return (newptr);
	}

	/* If the current block is the last block on the heap, then only extend
	 * the heap by the extra amount. */
	if ((size > oldsize) && ((FTRP(ptr) + DSIZE) == mem_sbrk(0))) {
		void *extra;
		size_t asize = ALIGNMENT *
		    ((size + DSIZE + (ALIGNMENT - 1)) / ALIGNMENT);
		if ((extra = mem_sbrk(asize - oldsize - DSIZE)) == (void *)-1)
			return (NULL);

		/* Initialize free block header/footer and the epilogue header.
		 */
		PUT(HDRP(ptr),
		    PACK((asize),
			1)); /* Update header of current block with new size. */
		PUT(FTRP(ptr),
		    PACK((asize), 1)); /* Update new footer of current block
					  with new size. */
		PUT(HDRP(NEXT_BLKP(ptr)), PACK(0, 1)); /* New epilogue header */

		return (ptr);
	}

	newptr = mm_malloc(size);

	/* If realloc() fails, the original block is left untouched.  */
	if (newptr == NULL)
		return (NULL);

	/* Copy just the old data, not the old header and footer. */
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

	/* Remove the prev/next free block that bp coalesces with from free
	 * list. Insert new coalesced free block into free list. */
	if (prev_alloc && next_alloc) { /* Case 1: none free. */
		ins_flist(bp, size);
		return (bp);
	} else if (prev_alloc &&
	    !next_alloc) { /* Case 2: next block is free. */
		rem_flist(NEXT_BLKP(bp));
		size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
		PUT(HDRP(bp), size);
		PUT(FTRP(bp), size);
		ins_flist(bp, size);
	} else if (!prev_alloc &&
	    next_alloc) { /* Case 3: prev block is free. */
		size += GET_SIZE(HDRP(PREV_BLKP(bp)));
		PUT(FTRP(bp), size);
		PUT(HDRP(PREV_BLKP(bp)), size);
		bp = PREV_BLKP(bp);
	} else { /* Case 4: both prev and next blocks are free. */
		rem_flist(NEXT_BLKP(bp));
		size += GET_SIZE(HDRP(PREV_BLKP(bp))) +
		    GET_SIZE(FTRP(NEXT_BLKP(bp)));
		PUT(HDRP(PREV_BLKP(bp)), size);
		PUT(FTRP(NEXT_BLKP(bp)), size);
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
	PUT(HDRP(bp), size);	     /* Free block header */
	PUT(FTRP(bp), size);	     /* Free block footer */
	PUT(HDRP(NEXT_BLKP(bp)), 1); /* New epilogue header */

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

	/* Find the correct size bin and traverse through to find the first fit
	 * and return a pointer to it. */
	fblock *bin = size_bin(asize);

	for (bp = bin->next; (bp != (void *)bin); bp = ((fblock *)bp)->next) {
		if (GET_SIZE(HDRP(bp)) >= asize) {
			break;
		}
	}

	if (bp != (void *)bin) {
		return bp;
	} else {
		while (bin < (flist + (NUM_BINS - 1))) {
			/* If can't find fit in this bin, check next larger size
			 * bin. */
			bin += 1;

			for (bp = bin->next; (bp != (void *)bin);
			     bp = ((fblock *)bp)->next) {
				if (GET_SIZE(HDRP(bp)) >= asize) {
					break;
				}
			}
			if (bp != (void *)bin)
				return bp;
		}
		/* No fit was found. */
		return (NULL);
	}
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
	size_t remain = csize - asize;
	rem_flist(bp);

	/* Splits the block if the remaining block is at least minimum block
	 * size, otherwise allocate the whole block. */
	if (remain >= MIN_BLK_SIZE) {
		PUT(HDRP(bp), PACK(asize, 1));
		PUT(FTRP(bp), PACK(asize, 1));
		bp = NEXT_BLKP(bp);
		PUT(HDRP(bp), PACK(remain, 0));
		PUT(FTRP(bp), PACK(remain, 0));
		/* Insert the remaining block back into free list. */
		ins_flist(bp, remain);
	} else {
		PUT(HDRP(bp), PACK(csize, 1));
		PUT(FTRP(bp), PACK(csize, 1));
	}
}

/*
 * Insert a free block into free list.
 *
 * @param bp the address of the free block.
 * @param size the size of the free block.
 */
static void
ins_flist(void *bp, size_t size)
{
	/* Find the correct segregated list and insert the free block to the end
	 * of it. */
	fblock *bin = size_bin(size);

	fblock *bin_prev = bin->prev;
	bin_prev->next = (fblock *)bp;
	bin->prev = (fblock *)bp;
	((fblock *)bp)->prev = bin_prev;
	((fblock *)bp)->next = bin;
}

/*
 * Remove a block from explicit free list.
 *
 * @param bp the address of the block.
 */
static void
rem_flist(void *bp)
{
	/* Make the block's previous and next pointers point to each other. */
	((fblock *)bp)->prev->next = ((fblock *)bp)->next;
	((fblock *)bp)->next->prev = ((fblock *)bp)->prev;
}

/*
 * Find the size bin corresponding to given size.
 *
 * @param size the size of the given block.
 * @return the address to the correct segregated list head.
 */
static fblock *
size_bin(size_t size)
{
	int bin;

	if (size <= 32) /* Bin 1: 2^5 */
		bin = 0;
	else if (size <= 128) /* Bin 2: 2^7 */
		bin = 1;
	else if (size <= 512) /* Bin 3: 2^9 */
		bin = 2;
	else if (size <= 2048) /* Bin 4: 2^11 */
		bin = 3;
	else if (size <= 16384) /* Bin 5: 2^14 */
		bin = 4;
	else /* Bin 6: above */
		bin = 5;

	return (flist + bin);
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
	/* Check if the block is 8-byte aligned properly. */
	if ((uintptr_t)bp % ALIGNMENT)
		printf("Error: %p is not doubleword aligned.\n", bp);

	/* Check if header matches footer, including if block size and
	 * allocation bit are correct. */
	if (GET(HDRP(bp)) != GET(FTRP(bp)))
		printf("Error: header does not match footer.\n");

	/* Check if the pointers in a heap block point to valid heap addresses.
	 */
	if (bp == NULL || bp < (void *)heap_listp || bp > mem_sbrk(0))
		printf(
		    "Error: block pointer does not point to a valid heap address.\n");

	if (GET_ALLOC(HDRP(bp))) {
		/* Check if any allocated blocks overlap. */
		if (GET_ALLOC(HDRP(NEXT_BLKP(bp))) &&
		    (HDRP(NEXT_BLKP(bp)) < FTRP(bp)))
			printf("Error: allocated blocks overlap.\n");
	} else {
		/* If the block is free, check the previous and next pointers
		 * and check for coalescing. */
		fblock *prev = ((fblock *)bp)->prev;
		fblock *next = ((fblock *)bp)->next;

		/* Check if every free block is actually in the free list. Check
		 * if the previous and next pointers point to valid free blocks.
		 */
		if (prev == NULL || GET_ALLOC(HDRP(prev)))
			printf(
			    "Error: previous pointer does not point to a valid free block.\n");
		if (next == NULL || GET_ALLOC(HDRP(next)))
			printf(
			    "Error: pointer does not point to a valid free block.\n");

		/* Check if there is any coalescing possible but has not yet
		 * been done. */
		if (!GET_ALLOC(HDRP(NEXT_BLKP(bp))) ||
		    !GET_ALLOC(HDRP(PREV_BLKP(bp)))) {
			printf(
			    "Error: can coalesce with an adjacent free block but has not yet done so.\n");
		}
	}
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

	/* Check all segregated lists in the explicit free list: if all the
	 * blocks are in the correct size bin. */
	size_t bin_sizes[NUM_BINS] = { 0, 32, 128, 512, 2048, 16384 };
	size_t bin_index = 0; /* Keep track of the index of the current bin. */
	for (fblock *currlist = flist; currlist < flist + NUM_BINS;
	     currlist += 1) {
		fblock *currblock = currlist->next;
		while (currblock != currlist) {
			/* Check if every block in the free list is marked as
			 * free. */
			if (GET_ALLOC(HDRP(currblock)))
				printf(
				    "Error: free block is not marked as free.\n");
			/* Check if every block is in the correct size bin. */
			size_t size = GET_SIZE(HDRP(currblock));
			if (bin_index == NUM_BINS - 1) {
				if (size <= bin_sizes[bin_index])
					printf(
					    "Error: free block is not in correct size bin.\n");
			} else {
				if (size <= bin_sizes[bin_index] ||
				    size > bin_sizes[bin_index + 1])
					printf(
					    "Error: free block is not in correct size bin.\n");
			}
			currblock = currblock->next;
		}
		bin_index++;
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
